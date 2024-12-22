import math
from typing import Callable

import torch
import torch.nn.functional as F


class PartialFC_SingleGPU(torch.nn.Module):
    """
    Single-GPU (non-distributed) equivalent of PartialFC_V2.
    It uses the same sampling logic but does not require torch.distributed.
    """

    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
    ):
        """
        Parameters:
        -----------
        margin_loss : Callable
            A callable that applies margin-based modification to logits (e.g., ArcFace margin).
        embedding_size : int
            The dimension of embedding.
        num_classes : int
            Total number of classes.
        sample_rate : float
            The rate of negative centers participating in the calculation, default is 1.0.
        fp16 : bool
            Whether to enable autocast with mixed precision.
        """
        super().__init__()
        self.margin_softmax = margin_loss
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.fp16 = fp16

        # The full weight matrix: [num_classes, embedding_size]
        self.weight = torch.nn.Parameter(
            torch.normal(mean=0.0, std=0.01, size=(num_classes, embedding_size))
        )

        # For sampling
        self.weight_index = None  # holds the indices of sampled classes

    def sample(self, labels: torch.Tensor, index_positive: torch.Tensor) -> torch.Tensor:
        """
        Subsample negative classes while keeping all positive classes.

        Modifies `labels` in-place so that valid labels match the subsampled weight row indices,
        and invalid (unselected) labels become -1.
        """
        with torch.no_grad():
            # Unique positive class indices in the current batch
            positive_classes = torch.unique(labels[index_positive], sorted=True)

            # Calculate how many classes we should sample
            num_sample = int(self.sample_rate * self.num_classes)

            # If we still have room after including all positives, randomly sample the rest
            if num_sample - positive_classes.size(0) > 0:
                perm = torch.rand(self.num_classes, device=labels.device)
                # push positives to the top so they definitely get included
                perm[positive_classes] = 2.0
                # pick top `num_sample` classes
                index = torch.topk(perm, k=num_sample, dim=0)[1]
                index = torch.sort(index)[0]
            else:
                # if num_sample < number of positive classes, just pick the positives
                index = positive_classes

            self.weight_index = index

            # Adjust label values so that they map into [0, len(index)-1]
            # for the newly sampled weight matrix
            labels[index_positive] = torch.searchsorted(index, labels[index_positive])

        # Return only the sampled rows of self.weight
        return self.weight[index]

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single-GPU forward pass.

        Parameters:
        -----------
        local_embeddings : torch.Tensor
            Embeddings for the current batch: [batch_size, embedding_size]
        local_labels : torch.Tensor
            Labels for the current batch: [batch_size]

        Returns:
        --------
        torch.Tensor
            Scalar loss (cross-entropy).
        """
        # Ensure correct shape/type
        local_labels = local_labels.long().view(-1)

        # Mark which labels are in valid range
        index_positive = (local_labels >= 0) & (local_labels < self.num_classes)

        # Subsample if needed
        if self.sample_rate < 1.0:
            weight = self.sample(local_labels, index_positive)
        else:
            weight = self.weight
            self.weight_index = None

        device_type = local_embeddings.device.type
        # Normalize embeddings & weights
        with torch.amp.autocast(enabled=self.fp16, device_type=device_type):
            norm_embeddings = F.normalize(local_embeddings, dim=1)
            norm_weight = F.normalize(weight, dim=1)
            logits = F.linear(norm_embeddings, norm_weight)

        original_logits = logits.clone()

        # Clamp for numerical stability
        logits = logits.clamp(-1, 1)

        # Apply margin-based softmax (e.g. ArcFace margin)
        logits = self.margin_softmax(logits, local_labels)

        # Standard cross-entropy over valid samples
        valid_mask = (local_labels != -1)
        if valid_mask.any():
            loss = F.cross_entropy(
                logits[valid_mask],
                local_labels[valid_mask],
            )
            logits = logits[valid_mask]
        else:
            # If no valid samples, return zero
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        modified_logits = logits.clone()

        return loss, modified_logits, original_logits # Don't use modified_logits to predict the class, it is only for loss calculation. You should compute the class using the

import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedLinearHeadWithCombinedMargin(nn.Module):
    """
    Linear head that:
      1. Normalizes both embeddings and weights.
      2. Performs a matrix multiply to get raw cos(theta) logits.
      3. Feeds logits + labels into CombinedMarginLoss to apply any margin manipulations.
    """
    def __init__(self, embedding_size, num_classes, margin_loss):
        """
        Args:
            embedding_size (int): Dimension of the incoming features.
            num_classes (int): Number of classes for classification.
            margin_loss (CombinedMarginLoss): Your combined margin loss module.
        """
        super().__init__()
        # A trainable weight matrix with shape [num_classes, embedding_size].
        # Typically, we do NOT use bias for ArcFace/CosFace style heads.
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        # The CombinedMarginLoss instance you provided
        self.margin_loss = margin_loss

    def forward(self, embeddings, labels=None):
        """
        Args:
            embeddings: [batch_size, embedding_size]
            labels: [batch_size]  (optional; if None, we just return raw logits)

        Returns:
            logits if labels=None, or margin-adjusted logits if labels is not None.
            Typically, you'll pass these logits into F.cross_entropy(logits, labels).
        """
        # 1) Normalize embeddings
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)  # [B, D]

        # 2) Normalize the weight matrix
        normalized_weight = F.normalize(self.weight, p=2, dim=1)     # [C, D]

        # 3) Dot product to get cos(theta) = x_norm · w_norm^T
        #    equivalent to: logits = normalized_embeddings @ normalized_weight.T
        logits = F.linear(normalized_embeddings, normalized_weight)  # [B, C]
        original_logits = logits.clone()

        # 4) If labels are given, apply your CombinedMarginLoss to modify the logits
        if labels is not None:
            logits = self.margin_loss(logits, labels)

        # 5) Return either raw cos(theta) or margin-modified logits
        return logits, original_logits

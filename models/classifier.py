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

        # Normalize embeddings & weights
        with torch.amp.autocast(enabled=self.fp16):
            norm_embeddings = F.normalize(local_embeddings, dim=1)
            norm_weight = F.normalize(weight, dim=1)
            logits = F.linear(norm_embeddings, norm_weight)

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

        return loss, logits

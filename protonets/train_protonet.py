import logging
import os
import sys
import time
from torch.nn import functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import random
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
from dataset.face_identity_dataset import FaceDetectAlign, IdentityImageDataset
from torchvision import transforms
from utils.utils import time_load_dataset, time_load_meta_dataset, time_load_folded_dataset

from dataset import root_datasets
from models import CamileNet
from config import parse_args
import wandb


# Create a face detection + alignment transform
face_detect_align = FaceDetectAlign(
    detector=None,  # Let it auto-create MTCNN if installed
    output_size=(112, 112),
    box_enlarge=1.5  # Enlarge bounding box slightly
)

# Compose with other transforms, e.g. ToTensor
transform_pipeline = transforms.Compose([
    face_detect_align,
    transforms.ToTensor()
])

def pairwise_distance_logits(query, support):
    """
    Compute the pairwise distance between query embeddings and support embeddings.
    Args:
        query: Tensor of shape (n_queries, d) containing the query embeddings.
        support: Tensor of shape (n_support, d) containing the support embeddings.
    Returns:
        distances: Tensor of shape (n_queries, n_support) containing the pairwise distances.
    """
    n_queries = query.size(0)
    n_support = support.size(0)
    d = query.size(1)
    query = query.unsqueeze(1).expand(n_queries, n_support, d)
    support = support.unsqueeze(0).expand(n_queries, n_support, d)
    distances = torch.pow(query - support, 2).sum(2)
    return -distances


def accuracy(preds, targets):
    """
    Compute the accuracy of predictions.
    Args:
        preds: Tensor of shape (n_queries, n_support) containing the predicted logits.
        targets: Tensor of shape (n_queries) containing the true labels.
    Returns:
        accuracy: Tensor containing the accuracy of the predictions.
    """
    predictions = preds.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(
    model,
    batch,
    ways,
    shots,
    shots_query,
    metric=None,
    device=None,
    max_batch_size=None,
    allow_nograd=False,
):
    """
    Adapt the model to a new task in a Prototypical Networks style.
    Args:
        model: Feature-extractor that produces embeddings. (NOT a MAML-learner)
        batch: (data, labels)
        ways: Number of classes in the task.
        shots: Number of support examples per class in the task.
        shots_query: Number of query examples per class in the task.
        metric: Distance metric function (query, support) -> logits.
        device: Device on which to run the computation.
        max_batch_size: If not None, we pass the data through the feature extractor in chunks.
        allow_nograd: If True, wrap forward passes in torch.no_grad() (for validation).
    Returns:
        loss: Cross-entropy loss.
        acc: Accuracy on the query set.
    """
    if metric is None:
        metric = pairwise_distance_logits  # e.g. a Euclidean or negative L2 distance function

    if device is None:
        device = torch.device("cpu")

    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Remove batch dimension (if it's shape (1, ways*(shots+query), C, H, W))
    # Sometimes data is shaped [1, total_samples, ...], so we squeeze(0).
    data = data.squeeze(0)
    labels = labels.squeeze(0)

    # Sort the samples by label so we can easily slice out support/query
    sort_indices = torch.sort(labels).indices
    data = data[sort_indices]
    labels = labels[sort_indices]

    # Compute embeddings, in chunks if max_batch_size is set
    # -----------------------------------------------------------------
    def chunked_feature_extraction(x, model, chunk_size):
        """
        Split 'x' along first dimension in chunks of size 'chunk_size',
        run through model, then re-cat results.
        """
        if chunk_size is None:
            return model(x)
        # x.shape = [total_samples, C, H, W]
        num_samples = x.shape[0]
        embeddings_list = []
        i = 0
        while i < num_samples:
            end = min(i + chunk_size, num_samples)
            with torch.set_grad_enabled(not allow_nograd):
                # If allow_nograd=True, we do no_grad in the outer scope.
                emb_chunk = model(x[i:end])
            embeddings_list.append(emb_chunk)
            i = end
        return torch.cat(embeddings_list, dim=0)

    if allow_nograd:
        with torch.no_grad():
            embeddings = chunked_feature_extraction(data, model, max_batch_size)
    else:
        embeddings = chunked_feature_extraction(data, model, max_batch_size)
    # -----------------------------------------------------------------

    total_per_class = shots + shots_query
    # We expect data.shape[0] == ways * (shots + shots_query)
    assert (
        embeddings.shape[0] == ways * total_per_class
    ), f"Mismatch: embeddings have {embeddings.shape[0]}, expected {ways * total_per_class}"

    # Build support/query masks
    support_mask = np.zeros(embeddings.size(0), dtype=bool)
    # For each class, pick 'shots' support from the top
    selection = np.arange(ways) * total_per_class
    for offset in range(shots):
        support_mask[selection + offset] = True

    support_mask = torch.from_numpy(support_mask)
    query_mask = ~support_mask

    # Slice out support & query
    support_embeddings = embeddings[support_mask]
    query_embeddings = embeddings[query_mask]
    query_labels = labels[query_mask].long()

    # Compute class prototypes by averaging support embeddings per class
    # support_embeddings.shape = [ways * shots, d]
    # We chunk them into (ways, shots, d) => mean over axis=1 => (ways, d)
    support_embeddings = support_embeddings.view(ways, shots, -1)
    prototypes = support_embeddings.mean(dim=1)  # shape (ways, d)

    # Evaluate distance-based classification: distances = metric(query_embeddings, prototypes)
    # distances.shape => [num_query, ways]
    distances = metric(query_embeddings, prototypes)

    # Cross-entropy loss & accuracy
    loss = F.cross_entropy(distances, query_labels)
    acc = accuracy(distances, query_labels)

    return loss, acc


def main(
    ways=5,
    shots=5,
    shots_query=5,
    meta_learning_rate=0.001,
    meta_batch_size=32,
    max_batch_size=None,
    iterations=1000,
    use_cuda=1,
    seed=42,
    number_train_tasks=-1,
    number_valid_tasks=-1,
    number_test_tasks=-1,
    patience=10,
    save_path="checkpoint/checkpoint.pth",
    debug_mode=False,
    use_wandb=False,
    network="camilenet",
    embedding_size=512,
    resume_from_checkpoint=False,
    run_str="run_without_script",
):
    # 1) Create a session string from the current date/time
    session_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    # 2) Use that session string in the log file name
    log_filename = f"{run_str}_{session_time}.log"

    # 3) Configure logging (including our session info in the format)
    logging.basicConfig(
        filename=log_filename,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    config = {
        "meta_learning_rate": meta_learning_rate,
        "meta_batch_size": meta_batch_size,
        "max_batch_size": max_batch_size,
        "iterations": iterations,
        "number_train_tasks": number_train_tasks,
        "number_valid_tasks": number_valid_tasks,
        "number_test_tasks": number_test_tasks,
        "patience": patience,
        "debug_mode": debug_mode,
        "network": network,
        "embedding_size": embedding_size,
        "resume_from_checkpoint": resume_from_checkpoint,
    }
    logging.info(f"Configuration: {config}")
    if use_wandb:
        wandb.init(
            project="edgeface-maml-anil",
            entity="benchmark_bros",
            config=config,
        )

    use_cuda = bool(use_cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")

    # Load datasets
    casiawebface_dataset = time_load_dataset(
        root_datasets.CASIA_WEB_FACE_ROOT,
        transform_pipeline,
        shots + shots_query,
        logging=logging,
    )
    age30_dataset = time_load_dataset(
        root_datasets.AGEDB_30_ROOT, transform_pipeline, shots + shots_query, logging=logging
    )
    bupt_dataset = time_load_dataset(
        root_datasets.BUPT_CBFACE_ROOT, transform_pipeline, shots + shots_query, logging=logging
    )
    ca_lfw_dataset = time_load_dataset(
        root_datasets.CA_LFW_ROOT, transform_pipeline, shots + shots_query, logging=logging
    )
    # cfp_fp_dataset = time_load_dataset(
    #     root_datasets.CFP_FP_ROOT,
    #     transform_pipeline,
    #     shots + shots_query,
    #     logging=logging
    # ) # CURRENTLY BROKEN DUE TO FILESTRUCTURE HAVING TWO FOLDERS - FIX IT
    cp_lfw_dataset = time_load_dataset(
        root_datasets.CP_LFW_ROOT, transform_pipeline, shots + shots_query, logging=logging
    )
    ijbb_dataset = time_load_dataset(
        root_datasets.IJBB_ROOT, transform_pipeline, shots + shots_query, logging=logging
    )
    ijbc_dataset = time_load_dataset(
        root_datasets.IJBC_ROOT, transform_pipeline, shots + shots_query, logging=logging
    )
    lfw_dataset = time_load_dataset(
        root_datasets.LFW_ROOT, transform_pipeline, shots + shots_query, logging=logging
    )
    ms1mv2_datasets = time_load_folded_dataset(
        root_datasets.MS1MV2_ROOT, transform_pipeline, shots + shots_query, logging=logging
    )
    umdfaces_dataset = time_load_dataset(
        root_datasets.UMDFACES_ROOT, transform_pipeline, shots + shots_query, logging=logging
    )
    glint360_datasets = time_load_folded_dataset(
        root_datasets.GLINT360K_ROOT, transform_pipeline, shots + shots_query, logging=logging
    )

    # Load meta-datasets
    casiawebface_metadataset = time_load_meta_dataset(
        casiawebface_dataset, logging=logging
    )
    age30_metadataset = time_load_meta_dataset(age30_dataset, logging=logging)
    bupt_metadataset = time_load_meta_dataset(bupt_dataset, logging=logging)
    ca_lfw_metadataset = time_load_meta_dataset(ca_lfw_dataset, logging=logging)
    # cfp_fp_metadataset = time_load_meta_dataset(cfp_fp_dataset, logging=logging)  # CURRENTLY BROKEN DUE TO FILESTRUCTURE HAVING TWO FOLDERS - FIX IT
    cp_lfw_metadataset = time_load_meta_dataset(cp_lfw_dataset, logging=logging)
    ijbb_metadataset = time_load_meta_dataset(ijbb_dataset, logging=logging)
    ijbc_metadataset = time_load_meta_dataset(ijbc_dataset, logging=logging)
    lfw_metadataset = time_load_meta_dataset(lfw_dataset, logging=logging)
    ms1mv2_metadatasets = [
        time_load_meta_dataset(ms1mv2_dataset, logging=logging)
        for ms1mv2_dataset in ms1mv2_datasets
    ]
    umdfaces_metadataset = time_load_meta_dataset(umdfaces_dataset, logging=logging)
    glint360_metadatasets = [
        time_load_meta_dataset(glint360_dataset, logging=logging)
        for glint360_dataset in glint360_datasets
    ]

    # Create list of datasets to be used
    train_datasets = [casiawebface_metadataset, bupt_metadataset, umdfaces_metadataset]
    train_datasets.extend(ms1mv2_metadatasets)
    train_datasets.extend(glint360_metadatasets)
    valid_datasets = [
        age30_metadataset,
        ca_lfw_metadataset,
        cp_lfw_metadataset,
        ijbb_metadataset,
        ijbc_metadataset,
        lfw_metadataset,
    ]  # Missing CFP-FP due to broken dataset

    train_tasksets = []
    train_tasksets_identity_size = []
    for dataset in train_datasets:
        identity_size = len(dataset)
        logging.info(f"Number of samples in {dataset}: {identity_size}")
        train_tasksets_identity_size.append(identity_size)
        train_taskset = l2l.data.TaskDataset(
            dataset,
            task_transforms=[
                FusedNWaysKShots(dataset, n=ways, k=shots + shots_query),
                LoadData(dataset),
                RemapLabels(dataset),
                ConsecutiveLabels(dataset),
            ],
            num_tasks=number_valid_tasks if not debug_mode else 50,
        )
        train_tasksets.append(train_taskset)

        logging.info(f"Loaded training taskset for {dataset}")

    valid_tasksets = []
    valid_tasksets_identity_size = []
    for dataset in valid_datasets:
        identity_size = len(dataset)
        logging.info(f"Number of identities in {dataset}: {identity_size}")
        valid_tasksets_identity_size.append(identity_size)

        valid_taskset = l2l.data.TaskDataset(
            dataset,
            task_transforms=[
                FusedNWaysKShots(dataset, n=ways, k=shots + shots_query),
                LoadData(dataset),
                RemapLabels(dataset),
                ConsecutiveLabels(dataset),
            ],
            num_tasks=number_valid_tasks if not debug_mode else 50,
        )
        valid_tasksets.append(valid_taskset)

        logging.info(f"Loaded validation taskset for {dataset}")

    prob_train = [
        identity_size / sum(train_tasksets_identity_size)
        for identity_size in train_tasksets_identity_size
    ]
    prob_valid = [
        identity_size / sum(valid_tasksets_identity_size)
        for identity_size in valid_tasksets_identity_size
    ]

    if network == "camilenet":
        model = CamileNet(
            input_channels=3,
            hidden_size=64,
            embedding_size=embedding_size,
            output_size=ways
        )
    else:
        raise ValueError(f"Unknown network: {network}")
    feature_extractor = model.features
    feature_extractor.to(device)

    all_parameters = list(feature_extractor.parameters())
    num_params = sum([np.prod(p.size()) for p in all_parameters])
    logging.info(f"Total number of parameters: {num_params / 1e6:.2f}M")

    optimizer = torch.optim.Adam(all_parameters, lr=meta_learning_rate, weight_decay=1e-4)

    # Make sure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if resume_from_checkpoint:
        checkpoint = torch.load(save_path)
        resume_epoch = checkpoint['epoch']
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info(f"Resuming training from epoch {resume_epoch}")
    else:
        resume_epoch = 0

    best_meta_val_error = float('inf')
    patience_counter = 0

    iteration = resume_epoch
    iterations += resume_epoch

    # Log session start info:
    logging.info(f"Session started at: {session_time}")
    logging.info(f"Initial iteration: {iteration} (Running until {iterations - 1})")

    for iteration in range(resume_epoch, iterations):
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0

        feature_extractor.train()
        # Meta-train steps
        for _ in range(meta_batch_size):
            # Sample from one of the training tasksets
            train_tasks = random.choices(train_tasksets, weights=prob_train, k=1)[0]
            batch = train_tasks.sample()
            loss, acc = fast_adapt(
                feature_extractor, batch, ways, shots, shots_query, device=device, max_batch_size=max_batch_size, allow_nograd=False
            )
            loss.backward()
            meta_train_error += loss.item()
            meta_train_accuracy += acc.item()

        avg_train_error = meta_train_error / meta_batch_size
        avg_train_accuracy = meta_train_accuracy / meta_batch_size

        logging.info(f"\nIteration: {iteration} / {iterations}")
        logging.info(f"Meta Train Loss: {avg_train_error:.4f}")
        logging.info(f"Meta Train Accuracy: {avg_train_accuracy:.4f}")

        # Average gradients
        for p in all_parameters:
            if p.grad is not None:
                p.grad.data.mul_(1.0 / meta_batch_size)
        optimizer.step()

        # Evaluate on Meta-Validation tasks for early stopping
        optimizer.zero_grad()
        # Free GPU memory
        torch.cuda.empty_cache()
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        feature_extractor.eval()
        for _ in range(meta_batch_size):
            # Sample from one of the validation tasksets using the weighted probability
            valid_tasks = random.choices(valid_tasksets, weights=prob_valid, k=1)[0]
            batch = valid_tasks.sample()
            loss, acc = fast_adapt(
                feature_extractor, batch, ways, shots, shots_query, device=device, max_batch_size=max_batch_size, allow_nograd=True
            )
            meta_valid_error += loss.item()
            meta_valid_accuracy += acc.item()

            # Free the gradients
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        meta_valid_error /= meta_batch_size
        meta_valid_accuracy /= meta_batch_size

        logging.info(f"Meta Val Loss: {meta_valid_error:.4f}")
        logging.info(f"Meta Val Accuracy: {meta_valid_accuracy:.4f}")

        if use_wandb:
            wandb.log({
                "meta_train_loss": avg_train_error,
                "meta_train_accuracy": avg_train_accuracy,
                "meta_val_loss": meta_valid_error,
                "meta_val_accuracy": meta_valid_accuracy,
            })

        # Early stopping logic
        if meta_valid_error < best_meta_val_error:
            logging.info(
                f"New best meta-val error "
                f"({best_meta_val_error:.4f} -> {meta_valid_error:.4f}). "
                f"Saving checkpoint."
            )
            best_meta_val_error = meta_valid_error
            patience_counter = 0

            checkpoint = {
                'epoch': iteration,
                'feature_extractor': feature_extractor.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, save_path)

        else:
            patience_counter += 1
            logging.info(f"No improvement in meta-val loss. Patience: {patience_counter}")
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered. No improvement for {patience} iterations.")
                break

if __name__ == '__main__':
    options = parse_args()
    main(
        ways=options.ways,
        shots=options.shots,
        shots_query=options.shots_query,
        meta_learning_rate=options.meta_learning_rate,
        meta_batch_size=options.meta_batch_size,
        max_batch_size=options.max_batch_size,
        iterations=options.iterations,
        use_cuda=options.use_cuda,
        seed=options.seed,
        number_train_tasks=options.number_train_tasks,
        number_valid_tasks=options.number_valid_tasks,
        number_test_tasks=options.number_test_tasks,
        patience=options.patience,
        debug_mode=options.debug_mode,
        use_wandb=options.use_wandb,
        network=options.network,
        embedding_size=options.embedding_size,
        resume_from_checkpoint=options.resume_from_checkpoint,
        run_str=options.run_str,
    )

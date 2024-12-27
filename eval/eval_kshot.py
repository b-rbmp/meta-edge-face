import logging
import os
import sys
import time
import tqdm 

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# 1) Create a session string from the current date/time
session_time = time.strftime("%Y-%m-%d_%H-%M-%S")

# 2) Use that session string in the log file name
log_filename = f"eval_5way5shot_{session_time}.log"

# 3) Configure logging (including our session info in the format)
logging.basicConfig(
    filename=log_filename,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

import torch
import argparse
from dataclasses import dataclass
import numpy as np
import random
from dataset.face_identity_dataset import FaceDetectAlign, IdentityImageDataset
from torchvision import transforms
from dataset import root_datasets
from models import get_model
from models import CamileNet, CamileNet130k
from models.camilenet_v3 import ProtoNet


@dataclass
class FiveWayEvalConfig:
    """Configuration for 5-way, k-shot prototypical evaluation."""
    use_cuda: int
    seed: int
    network: str = "edgeface_xs_gamma_06"
    embedding_size: int = 512
    checkpoint_str: str = "checkpoint/checkpoint.pth"
    shots: int = 5
    queries: int = 5
    ways: int = 5
    num_episodes: int = 1000


def get_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for 5-way 5-shot evaluation.
    """
    parser = argparse.ArgumentParser(description="5-way 5-shot evaluation")
    parser.add_argument("--use-cuda", type=int, default=1, help="Use CUDA (1) or CPU (0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--network",
        type=str,
        default="camilenet",
        help="Network architecture to use (e.g. camilenet, edgeface_xs_gamma_06)",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=64,
        help="Size of the final embedding layer",
    )
    parser.add_argument(
        "--checkpoint_str",
        type=str,
        default="checkpoint/checkpoint.pth",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--shots", type=int, default=5, help="Number of support shots per class"
    )
    parser.add_argument(
        "--queries", type=int, default=5, help="Number of query examples per class"
    )
    parser.add_argument(
        "--ways", type=int, default=5, help="Number of classes in each episode"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=20, help="How many episodes to sample"
    )
    return parser


def parse_args() -> FiveWayEvalConfig:
    """
    Parse command-line arguments and return a configuration object.
    """
    parser = get_parser()
    args = parser.parse_args()
    return FiveWayEvalConfig(
        use_cuda=args.use_cuda,
        seed=args.seed,
        network=args.network,
        embedding_size=args.embedding_size,
        checkpoint_str=args.checkpoint_str,
        shots=args.shots,
        queries=args.queries,
        ways=args.ways,
        num_episodes=args.num_episodes,
    )


def time_load_dataset(root_dir, transform_pipeline, min_samples_per_identity):
    """
    Loads the dataset with the given transforms, ensuring each identity
    has at least 'min_samples_per_identity' samples.
    """
    time_start = time.time()
    dataset = IdentityImageDataset(
        root_dir=root_dir,
        transform=transform_pipeline,
        min_samples_per_identity=min_samples_per_identity,
    )
    time_end = time.time()
    logging.info(f"Time to load dataset {root_dir}: {time_end - time_start:.2f}s")
    return dataset


############################
# 5-way 5-shot sampling logic
############################

def sample_5way5shot_indices(dataset, ways=5, shots=5, queries=5):
    """
    From the dataset, sample 'ways' distinct classes (identities).
    From each class, sample (shots + queries) distinct images.

    Returns:
      (support_indices, query_indices, support_labels, query_labels)
      Each is a list of dataset indices or label IDs [0..ways-1].
      If the dataset doesn't have enough classes/images, returns (None, None, None, None).
    """
    # 1) Group dataset indices by label
    label_to_indices = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)

    # 2) Filter out labels that don't have enough images
    valid_labels = [
        lbl for lbl, idxs in label_to_indices.items()
        if len(idxs) >= (shots + queries)
    ]
    if len(valid_labels) < ways:
        # Not enough classes with sufficient images for a 5-way
        return None, None, None, None

    # 3) Sample 'ways' distinct labels
    chosen_labels = random.sample(valid_labels, ways)

    support_indices = []
    support_labels = []
    query_indices = []
    query_labels = []

    for lbl_idx, lbl in enumerate(chosen_labels):
        all_indices = label_to_indices[lbl]
        chosen = random.sample(all_indices, shots + queries)
        chosen_support = chosen[:shots]
        chosen_query = chosen[shots:]

        support_indices.extend(chosen_support)
        support_labels.extend([lbl_idx] * shots)
        query_indices.extend(chosen_query)
        query_labels.extend([lbl_idx] * queries)

    return support_indices, query_indices, support_labels, query_labels


def compute_embeddings(model, dataset, device, indices, batch_size=64):
    """
    Given a list of dataset indices, compute embeddings for those images in batches.
    Returns a torch.Tensor of shape [len(indices), embed_dim].
    """
    model.eval()
    all_embeds = []
    with torch.no_grad():
        start = 0
        while start < len(indices):
            end = min(start + batch_size, len(indices))
            batch_idxs = indices[start:end]
            images = []
            for idx in batch_idxs:
                img, _ = dataset[idx]
                images.append(img)
            images = torch.stack(images, dim=0).to(device)
            embeds = model(images)
            all_embeds.append(embeds.cpu())
            start = end
    return torch.cat(all_embeds, dim=0)


def proto_accuracy(model, dataset, device, ways=5, shots=5, queries=5, episodes=1000, batch_size=64):
    """
    Perform 5-way k-shot evaluation in a ProtoNet style:
      1) Sample 'ways' classes, with 'shots' support and 'queries' query images each.
      2) Compute prototypes by averaging the support embeddings for each class.
      3) Classify queries by nearest prototype (L2 distance).
      4) Repeat for 'episodes' times and compute overall mean/std accuracy.
    """
    accuracies = []
    for _ in tqdm.trange(episodes):
        s_indices, q_indices, s_labels, q_labels = sample_5way5shot_indices(
            dataset, ways=ways, shots=shots, queries=queries
        )
        # If the dataset can't provide enough classes/images, end
        if s_indices is None:
            break

        # Compute support + query embeddings
        support_embeds = compute_embeddings(model, dataset, device, s_indices, batch_size)
        query_embeds = compute_embeddings(model, dataset, device, q_indices, batch_size)

        # Reshape support to [ways, shots, embed_dim], then average to get prototypes
        support_embeds = support_embeds.view(ways, shots, -1)
        prototypes = support_embeds.mean(dim=1)  # [ways, embed_dim]

        # Compute L2 distance from each query to each prototype
        num_queries = ways * queries
        q_expand = query_embeds.unsqueeze(1).expand(num_queries, ways, prototypes.size(1))
        p_expand = prototypes.unsqueeze(0).expand(num_queries, ways, prototypes.size(1))
        dist = (q_expand - p_expand).pow(2).sum(dim=2)  # [num_queries, ways]

        # Predictions = nearest prototype
        preds = dist.argmin(dim=1)

        # Accuracy for this episode
        q_labels_tensor = torch.tensor(q_labels, dtype=torch.long)
        correct = (preds == q_labels_tensor).sum().item()
        acc_episode = correct / float(num_queries)
        accuracies.append(acc_episode)

    if len(accuracies) == 0:
        return 0.0, 0.0

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    return mean_acc, std_acc


def main(
    use_cuda=1,
    seed=42,
    network="edgeface_xs_gamma_06",
    embedding_size=512,
    checkpoint_str="checkpoint/checkpoint.pth",
    shots=1,
    queries=1,
    ways=2,
    num_episodes=10,
):
    logging.info("Starting 5-way k-shot eval script")
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

    # Minimal face-detection & alignment transforms for evaluation
    face_detect_align = FaceDetectAlign(
        detector=None,
        output_size=(112, 112),
        box_enlarge=1.3,
    )
    transform_pipeline = transforms.Compose([
        face_detect_align,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load datasets
    age30_dataset = time_load_dataset(root_datasets.AGEDB_30_ROOT, transform_pipeline, 3)
    lfw_dataset = time_load_dataset(root_datasets.LFW_ROOT, transform_pipeline, 3)
    # Add additional sets if needed
    eval_datasets = {
        "AgeDB30": age30_dataset,
        "LFW": lfw_dataset,
    }

    # Initialize model
    if network == "edgeface_xs_gamma_06":
        logging.info("Using EdgeFace XS model")
        feature_extractor = get_model(
            network, dropout=0.0, fp16=False, num_features=embedding_size
        )
    elif network == "camilenet":
        logging.info("Using CamileNet model")
        model = CamileNet(
            input_channels=3,
            hidden_size=embedding_size,
            embedding_size=embedding_size,
            output_size=10,  # Unused during inference
        )
        feature_extractor = model.features
    elif network == "camilenet_v3":
        logging.info("Using CamileNet v3 model")
        model = ProtoNet()
        feature_extractor = model
    elif network == "camilenet130k":
        logging.info("Using CamileNet130k model")
        model = CamileNet130k(
            input_channels=3,
            hidden_size=embedding_size,
            embedding_size=embedding_size,
            output_size=10,  # Unused during inference
        )
        feature_extractor = model.features
    else:
        raise ValueError(f"Unknown network: {network}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_str, map_location=device)
    feature_extractor.load_state_dict(checkpoint["feature_extractor"])
    feature_extractor.to(device)

    # Evaluate each dataset with 5-way k-shot
    metrics_data = {}
    for name, ds in eval_datasets.items():
        mean_acc, std_acc = proto_accuracy(
            feature_extractor,
            ds,
            device,
            ways=ways,
            shots=shots,
            queries=queries,
            episodes=num_episodes,
            batch_size=64,
        )
        logging.info(f"[{name}] {ways}-way {shots}-shot Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        metrics_data[name] = (mean_acc, std_acc)

    # Summarize
    for name, (acc_mean, acc_std) in metrics_data.items():
        logging.info(f"{name} -> {ways}-way {shots}-shot Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")


if __name__ == "__main__":
    options = parse_args()
    logging.info(f"Configuration: {options}")
    main(
        use_cuda=options.use_cuda,
        seed=options.seed,
        network=options.network,
        embedding_size=options.embedding_size,
        checkpoint_str=options.checkpoint_str,
        shots=options.shots,
        queries=options.queries,
        ways=options.ways,
        num_episodes=options.num_episodes,
    )

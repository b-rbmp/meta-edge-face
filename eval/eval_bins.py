import logging
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# 1) Create a session string from the current date/time
session_time = time.strftime("%Y-%m-%d_%H-%M-%S")

# 2) Use that session string in the log file name
log_filename = f"eval_bins_{session_time}.log"

# 3) Configure logging (including our session info in the format)
logging.basicConfig(
    filename=log_filename,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

import pickle
import numpy as np

np.bool = np.bool_
import mxnet as mx
from mxnet import nd

import torch
import argparse
from dataclasses import dataclass
import random
from dataset.face_identity_dataset import FaceDetectAlign, IdentityImageDataset
from torchvision import transforms
from collections import defaultdict
from dataset import root_datasets
from models import get_model
from maml_anil.config import parse_args
from models import CamileNet, CamileNet130k, CamileNetV3
import sklearn
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from scipy import interpolate


@dataclass
class MAMLEvalConfig:
    use_cuda: int
    seed: int
    network: str = "edgeface_xs_gamma_06"
    embedding_size: int = 512
    checkpoint_str: str = "checkpoint/checkpoint.pth"
    batch_size: int = 64
    nrof_folds: int = 10
    threshold_start: float = 0
    threshold_end: float = 4
    threshold_step: float = 0.01


def get_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for MAML-based few-shot learning training.
    """
    parser = argparse.ArgumentParser(description="MAML-based few-shot learning")
    parser.add_argument(
        "--use-cuda", type=int, default=1, help="Use CUDA (1) or CPU (0)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--network",
        type=str,
        default="camilenet",
        help="Network architecture to use - camilenet | edgeface_xs_gamma_06",
    )
    parser.add_argument(
        "--embedding_size", type=int, default=64, help="Size of the embedding layer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--nrof_folds", type=int, default=10, help="Number of folds for evaluation"
    )
    parser.add_argument(
        "--threshold_start",
        type=float,
        default=0,
        help="Start threshold for ROC evaluation",
    )
    parser.add_argument(
        "--threshold_end",
        type=float,
        default=4,
        help="End threshold for ROC evaluation",
    )
    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.01,
        help="Step size for ROC evaluation",
    )

    parser.add_argument(
        "--checkpoint_str",
        type=str,
        default="checkpoint/checkpoint.pth",
        help="Path to the checkpoint file",
    )
    return parser


def parse_args() -> MAMLEvalConfig:
    """
    Parse command-line arguments and return a configuration object.
    """
    parser = get_parser()
    args = parser.parse_args()

    return MAMLEvalConfig(
        use_cuda=args.use_cuda,
        seed=args.seed,
        network=args.network,
        embedding_size=args.embedding_size,
        checkpoint_str=args.checkpoint_str,
        batch_size=args.batch_size,
        nrof_folds=args.nrof_folds,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.threshold_step,
    )


class LFold:
    """Simple wrapper around KFold for n_splits>1, else no split."""

    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            # If only 1-fold, train=test=all
            return [(indices, indices)]


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame))
    )
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(
    thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0
):
    """Compute TPR, FPR, and accuracy (via KFold cross-validation)."""
    assert embeddings1.shape[0] == embeddings2.shape[0]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    # If no PCA, just do direct distance
    if pca == 0:
        diff = embeddings1 - embeddings2
        dist = np.sum(np.square(diff), 1)
        logging.info(
            f"Current bounds of distance (adjust threshold accordingly): {np.min(dist):.4f} - {np.max(dist):.4f}"
        )

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            # Fit PCA on train set
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = embed1 - embed2
            dist = np.sum(np.square(diff), 1)

        # Find best threshold on the training set
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set]
            )
        best_threshold_index = np.argmax(acc_train)

        # Evaluate TPR/FPR on the test set for all thresholds
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = (
                calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
            )

        # Finally, record accuracy at best threshold for this fold
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set]
        )

    # Average TPR/FPR across folds
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_val(
    thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10
):
    """Computes the validation rate (VAL) at given FAR target."""
    assert embeddings1.shape[0] == embeddings2.shape[0]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = embeddings1 - embeddings2
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Compute FAR on train
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set]
            )
        # Find threshold that yields the target FAR
        if np.max(far_train) >= far_target:
            unique_far_train, unique_indices = np.unique(far_train, return_index=True)
            unique_thresholds = thresholds[unique_indices]
            f = interpolate.interp1d(
                unique_far_train, unique_thresholds, kind="slinear"
            )
            threshold = f(far_target)
        else:
            threshold = 0.0

        # Evaluate VAL/FAR on test
        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set]
        )

    val_mean = np.mean(val)
    val_std = np.std(val)
    far_mean = np.mean(far)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(
    embeddings,
    actual_issame,
    nrof_folds=10,
    pca=0,
    threshold_start=0,
    threshold_end=4,
    threshold_step=0.01,
):
    """Full evaluation: TPR/FPR/Accuracy across thresholds + VAL/FAR."""
    thresholds = np.arange(threshold_start, threshold_end, threshold_step)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    tpr, fpr, accuracy = calculate_roc(
        thresholds,
        embeddings1,
        embeddings2,
        np.asarray(actual_issame),
        nrof_folds=nrof_folds,
        pca=pca,
    )
    # For val and far
    val, val_std, far = calculate_val(
        thresholds,
        embeddings1,
        embeddings2,
        np.asarray(actual_issame),
        far_target=1e-3,
        nrof_folds=nrof_folds,
    )
    return tpr, fpr, accuracy, val, val_std, far


def compute_pairwise_embeddings(
    model, dataset, device, indices1, indices2, batch_size=64
):
    model.eval()
    num_pairs = len(indices1)
    all_embeddings = []

    # We'll combine the two sets of indices in a single list
    # so we can process them in a single forward pass loop:
    # but we remember how to "unpack" them afterward.
    merged_indices = []
    for i in range(num_pairs):
        merged_indices.append(indices1[i])
        merged_indices.append(indices2[i])

    with torch.no_grad():
        start = 0
        while start < len(merged_indices):
            end = min(start + batch_size, len(merged_indices))
            batch_idxs = merged_indices[start:end]
            images = []
            for idx in batch_idxs:
                img, _ = dataset[idx]
                images.append(img)
            images = torch.stack(images, dim=0).to(device)

            # forward pass
            embeds = model(images)  # shape: [batch_size, embed_dim]
            all_embeddings.append(embeds.cpu())
            start = end

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    return all_embeddings  # shape = [2 * num_pairs, embed_dim]


@torch.no_grad()
def load_bin(path, image_size):
    try:
        with open(path, "rb") as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, "rb") as f:
            bins, issame_list = pickle.load(f, encoding="bytes")  # py3
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())

    # print(data_list[0].shape)
    return data_list, issame_list


def evaluate_bin_dataset(
    bin_path,
    dataset_name,
    model,
    device,
    batch_size=64,
    nrof_folds=10,
    threshold_start=0.0,
    threshold_end=4.0,
    threshold_step=0.01,
    image_size=(112, 112),
):
    """
    Evaluates a .bin file that contains (bins, issame_list),
    similar to LFW/CALFW/CFP-FP style test files.

    1) Loads the .bin using load_bin(...).
    2) Computes embeddings by passing all images (flip=0 and flip=1)
       through the model in batches.
    3) Combines the embeddings for flip=0 and flip=1.
    4) Runs the standard 'evaluate(...)' routine to get metrics.
    """

    # 1) Load the bin
    data_list, issame_list = load_bin(bin_path, image_size=image_size)
    # data_list is [tensor_flip0, tensor_flip1], each shape = (2*N, 3, H, W)

    # 2) Compute embeddings for each flip version
    embeddings_list = []
    for flip_idx, data_tensor in enumerate(data_list):
        embeddings = []
        # We'll process in batches
        start_idx = 0
        while start_idx < data_tensor.shape[0]:
            end_idx = min(start_idx + batch_size, data_tensor.shape[0])
            batch_data = data_tensor[start_idx:end_idx].to(device)

            # Normalize or scale if needed:
            # e.g., images = (batch_data / 255.0 - 0.5) / 0.5
            images = (batch_data / 255.0 - 0.5) / 0.5

            # Forward pass
            with torch.no_grad():
                batch_emb = model(images)  # [batch_size, embed_dim]
            embeddings.append(batch_emb.cpu().numpy())
            start_idx = end_idx

        # Combine into one (N*2, embed_dim)
        embeddings = np.concatenate(embeddings, axis=0)
        embeddings_list.append(embeddings)

    # embeddings_list[0] = flip=0, embeddings_list[1] = flip=1
    # Each is shape: (2 * num_pairs, embed_dim)

    # Option A: Use only flip=0
    # combined_embeddings = sklearn.preprocessing.normalize(embeddings_list[0])
    #
    # Option B: Sum flip=0 and flip=1, then normalize:
    combined_embeddings = embeddings_list[0] + embeddings_list[1]
    combined_embeddings = sklearn.preprocessing.normalize(combined_embeddings)

    # 3) Evaluate
    #    We have 2*N images => N pairs => the 'issame_list' parallels these pairs
    #    The standard evaluate() expects embeddings in shape (2N, D)
    #    and issame_list as booleans for each of the N pairs.
    tpr, fpr, accuracy, val, val_std, far = evaluate(
        combined_embeddings,
        issame_list,
        nrof_folds=nrof_folds,
        pca=0,  # or set your PCA dimension if needed
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        threshold_step=threshold_step,
    )
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)

    logging.info(
        f"{dataset_name} Accuracy: {acc_mean:.5f} ± {acc_std:.5f}, "
        f"VAL: {val:.5f} ± {val_std:.5f}, FAR: {far:.5f}"
    )

    return {
        "tpr": tpr,
        "fpr": fpr,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "val": val,
        "val_std": val_std,
        "far": far,
    }


def main(
    use_cuda=1,
    seed=42,
    network="edgeface_xs_gamma_06",
    embedding_size=512,
    checkpoint_str="checkpoint/checkpoint.pth",
    batch_size=64,
    nrof_folds=10,
    threshold_start=0,
    threshold_end=4,
    threshold_step=0.01,
):
    logging.info("Starting eval script")
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

    # Create a face detection + alignment transform
    face_detect_align = FaceDetectAlign(
        detector=None,  # Let it auto-create MTCNN if installed
        output_size=(112, 112),
        box_enlarge=1.3,
    )
    transform_pipeline = transforms.Compose([
        face_detect_align,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load datasets

    eval_datasets = {
        "AgeDB30": root_datasets.BIN_AGEDB_30,
        "CA-LFW": root_datasets.BIN_CA_LFW,
        "CFP-FP": root_datasets.BIN_CFP_FP,
        "CFP-FF": root_datasets.BIN_CFP_FF,
        "CP-LFW": root_datasets.BIN_CP_LFW,
        "LFW": root_datasets.BIN_LFW,
        "VGG2-FP": root_datasets.BIN_VGG_FP,
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
            output_size=10,  # Doesn't matter, will get only feature extractor
        )
        feature_extractor = model.features
    elif network == "camilenet130k":
        logging.info("Using CamileNet130k model")
        model = CamileNet130k(
            input_channels=3,
            hidden_size=embedding_size,
            embedding_size=embedding_size,
            output_size=10,  # Doesn't matter, will get only feature extractor
        )
        feature_extractor = model.features
    elif network == "camilenet_v3":
        logging.info("Using CamileNetV3 model")
        model = CamileNetV3(
            x_dim=3,
            hid_dim=embedding_size,
            z_dim=embedding_size,
        )
        feature_extractor = model.features
    else:
        raise ValueError(f"Unknown network: {network}")

    checkpoint = torch.load(checkpoint_str, map_location=device)
    feature_extractor.load_state_dict(checkpoint["feature_extractor"])
    feature_extractor.to(device)

    # Evaluate each dataset
    metrics_data = {}
    for name, bin_path in eval_datasets.items():
        results = evaluate_bin_dataset(
            bin_path=bin_path,
            dataset_name=name,
            model=feature_extractor,
            device=device,
            batch_size=batch_size,
            nrof_folds=nrof_folds,
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            threshold_step=threshold_step,
            image_size=(112, 112),
        )
        metrics_data[name] = results

    # accuracy per dataset:
    for name, res in metrics_data.items():
        logging.info(
            f"{name} -> Accuracy: {res['acc_mean']:.4f} ± {res['acc_std']:.4f}"
        )


# Accuracy: LFW, CPLFW, CALFW, CFP-FP, and AgeDB30
# True Accept Rate: IJB-C dataset, where they applied the true acceptance rate (TAR) at a false acceptance rate (FAR) of 10−4, denoted as TAR at FAR=10−4
if __name__ == "__main__":
    options = parse_args()
    logging.info(f"Configuration: {options}")
    main(
        use_cuda=options.use_cuda,
        seed=options.seed,
        network=options.network,
        embedding_size=options.embedding_size,
        checkpoint_str=options.checkpoint_str,
        batch_size=options.batch_size,
        nrof_folds=options.nrof_folds,
        threshold_start=options.threshold_start,
        threshold_end=options.threshold_end,
        threshold_step=options.threshold_step,
    )

import logging
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# 1) Create a session string from the current date/time
session_time = time.strftime("%Y-%m-%d_%H-%M-%S")

# 2) Use that session string in the log file name
log_filename = f"eval_{session_time}.log"

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
from collections import defaultdict
from dataset import root_datasets
from models import get_model
from maml_anil.config import parse_args
from models import CamileNet, CamileNet130k, CamileNetV3

from sklearn.model_selection import KFold
from scipy import interpolate

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA

@dataclass
class MAMLEvalConfig:
    use_cuda: int
    seed: int
    network: str = "edgeface_xs_gamma_06"
    embedding_size: int = 512
    checkpoint_str: str = "checkpoint/checkpoint.pth"
    num_pos_pairs: int = 3000
    num_neg_pairs: int = 3000
    batch_size: int = 64
    nrof_folds: int = 10
    threshold_start: float = 0
    threshold_end: float = 4
    threshold_step: float = 0.01
    visualize_pca: bool = False


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
        "--num_pos_pairs", type=int, default=3000, help="Number of positive pairs"
    )
    parser.add_argument(
        "--num_neg_pairs", type=int, default=3000, help="Number of negative pairs"
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
        default=10,
        help="Start threshold for ROC evaluation",
    )
    parser.add_argument(
        "--threshold_end",
        type=float,
        default=4000,
        help="End threshold for ROC evaluation",
    )
    parser.add_argument(
        "--threshold_step", type=float, default=1, help="Step size for ROC evaluation"
    )
    parser.add_argument(
        "--checkpoint_str",
        type=str,
        default="checkpoint/checkpoint.pth",
        help="Path to the checkpoint file",
    )
    parser.add_argument("--visualize_pca", action="store_true", default=False,
                        help="Visualize 30 pairs via PCA in 2D")
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
        num_pos_pairs=args.num_pos_pairs,
        num_neg_pairs=args.num_neg_pairs,
        batch_size=args.batch_size,
        nrof_folds=args.nrof_folds,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.threshold_step,
        visualize_pca=args.visualize_pca,
    )


def time_load_dataset(root_dir, transform_pipeline, min_samples_per_identity):
    time_start = time.time()
    dataset = IdentityImageDataset(
        root_dir=root_dir,
        transform=transform_pipeline,
        min_samples_per_identity=min_samples_per_identity,
    )
    time_end = time.time()
    logging.info(f"Time to load dataset {root_dir}: {time_end - time_start:.2f}s")
    return dataset


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
    """
    Returns (tpr, fpr, overall_accuracy).
    This does not separate same/different; we do that if needed.
    """
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
    thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10
):
    """
    Return:
      tpr, fpr, accuracy_folds, dist, best_threshold_indices
    """
    assert embeddings1.shape[0] == embeddings2.shape[0]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy_folds = np.zeros(nrof_folds, dtype=float)

    best_threshold_indices = np.zeros(nrof_folds, dtype=int)

    diff = embeddings1 - embeddings2
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find best threshold on train set
        best_acc = -1.0
        best_idx = 0
        for threshold_idx, thr in enumerate(thresholds):
            _, _, acc_train = calculate_accuracy(thr, dist[train_set], actual_issame[train_set])
            if acc_train > best_acc:
                best_acc = acc_train
                best_idx = threshold_idx

        best_threshold_indices[fold_idx] = best_idx

        # Evaluate TPR/FPR on test set for all thresholds
        for threshold_idx, thr in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = (
                calculate_accuracy(thr, dist[test_set], actual_issame[test_set])
            )

        # Evaluate test accuracy at best threshold
        thr_best = thresholds[best_idx]
        _, _, test_acc = calculate_accuracy(thr_best, dist[test_set], actual_issame[test_set])
        accuracy_folds[fold_idx] = test_acc

    tpr = np.mean(tprs, axis=0)
    fpr = np.mean(fprs, axis=0)
    return tpr, fpr, accuracy_folds, dist, best_threshold_indices


def calculate_val(
    thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10
):
    """Computes the validation rate (VAL) at a given FAR target."""
    assert embeddings1.shape[0] == embeddings2.shape[0]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    diff = embeddings1 - embeddings2
    dist = np.sum(np.square(diff), 1)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    thresholds = np.array(thresholds)

    indices = np.arange(nrof_pairs)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # compute FAR on train
        far_train = np.zeros(len(thresholds))
        for i, thr in enumerate(thresholds):
            _, far_train[i] = calculate_val_far(thr, dist[train_set], actual_issame[train_set])

        # find threshold that yields target FAR
        if np.max(far_train) >= far_target:
            unique_far_train, unique_indices = np.unique(far_train, return_index=True)
            unique_thresholds = thresholds[unique_indices]
            f = interpolate.interp1d(unique_far_train, unique_thresholds, kind="slinear")
            thr = f(far_target)
        else:
            thr = 0.0

        # evaluate VAL/FAR on test
        val[fold_idx], far[fold_idx] = calculate_val_far(thr, dist[test_set], actual_issame[test_set])

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

    val = float(true_accept) / float(n_same) if n_same > 0 else 0.0
    far = float(false_accept) / float(n_diff) if n_diff > 0 else 0.0
    return val, far


def evaluate(
    embeddings1,
    embeddings2,
    actual_issame,
    nrof_folds=10,
    threshold_start=0,
    threshold_end=4,
    threshold_step=0.01,
):
    """
    Full evaluation: TPR/FPR/Accuracy across thresholds + VAL/FAR + dist & best_threshold info
    """
    thresholds = np.arange(threshold_start, threshold_end, threshold_step)

    # 1) Normal ROC calc
    tpr, fpr, accuracy_folds, dist, best_threshold_indices = calculate_roc(
        thresholds, embeddings1, embeddings2, actual_issame, nrof_folds
    )

    # 2) VAL/FAR
    val, val_std, far = calculate_val(
        thresholds, embeddings1, embeddings2, actual_issame, far_target=1e-3, nrof_folds=nrof_folds
    )

    return tpr, fpr, accuracy_folds, val, val_std, far, dist, best_threshold_indices, thresholds


def build_pairs_from_dataset(dataset, num_pos_pairs=3000, num_neg_pairs=3000):
    label_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        _, label = dataset[i]
        label_to_indices[label].append(i)

    label_groups = list(label_to_indices.items())

    same_pairs = []
    for label, idxs in label_groups:
        if len(idxs) < 2:
            continue
        random.shuffle(idxs)
        for i in range(len(idxs) - 1):
            same_pairs.append((idxs[i], idxs[i + 1], True))
    same_pairs = random.sample(same_pairs, min(len(same_pairs), num_pos_pairs))

    diff_pairs = []
    count_diffs = 0
    while count_diffs < num_neg_pairs:
        la, idxsA = random.choice(label_groups)
        idxA = random.choice(idxsA)
        lb, idxsB = random.choice(label_groups)
        if lb == la:
            continue
        idxB = random.choice(idxsB)
        diff_pairs.append((idxA, idxB, False))
        count_diffs += 1

    all_pairs = same_pairs + diff_pairs
    random.shuffle(all_pairs)

    indices1 = [p[0] for p in all_pairs]
    indices2 = [p[1] for p in all_pairs]
    issame_list = [p[2] for p in all_pairs]

    return indices1, indices2, issame_list


def compute_pairwise_embeddings(
    model, dataset, device, indices1, indices2, batch_size=64
):
    """
    Returns:
        embeddings_1: shape [num_pairs, embedding_dim]
        embeddings_2: shape [num_pairs, embedding_dim]
    """
    model.eval()
    num_pairs = len(indices1)
    all_embeddings = []

    # We'll merge the two index lists into one big list: [idx1_0, idx2_0, idx1_1, idx2_1, ...].
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

            embeds = model(images)
            all_embeddings.append(embeds.cpu())
            start = end

    # shape => (2 * num_pairs, embed_dim)
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

    # i-th pair's first image => row 2*i, second => row 2*i + 1
    embeddings_1 = all_embeddings[0::2]  # even rows
    embeddings_2 = all_embeddings[1::2]  # odd rows

    return embeddings_1, embeddings_2


def evaluate_dataset(
    dataset,
    model,
    device,
    dataset_name="Unknown",
    num_pos_pairs=3000,
    num_neg_pairs=3000,
    batch_size=64,
    nrof_folds=10,
    threshold_start=0,
    threshold_end=4,
    threshold_step=0.01,
    visualize_pca=False,
):
    logging.info(f"Evaluating dataset: {dataset_name}")

    # 1) Build pairs
    indices1, indices2, issame_list = build_pairs_from_dataset(
        dataset, num_pos_pairs=num_pos_pairs, num_neg_pairs=num_neg_pairs
    )
    issame_list = np.array(issame_list, dtype=bool)

    # 2) Compute embeddings
    embeddings1, embeddings2 = compute_pairwise_embeddings(
        model, dataset, device, indices1, indices2, batch_size=batch_size
    )

    # 3) Evaluate
    (tpr,
     fpr,
     accuracy_folds,
     val,
     val_std,
     far,
     dist,
     best_threshold_indices,
     thresholds) = evaluate(
        embeddings1,
        embeddings2,
        issame_list,
        nrof_folds=nrof_folds,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        threshold_step=threshold_step,
    )

    acc_mean = np.mean(accuracy_folds)
    acc_std = np.std(accuracy_folds)

    logging.info(
        f"[{dataset_name}] Overall Accuracy: {acc_mean:.5f} ± {acc_std:.5f}, "
        f"VAL: {val:.5f} ± {val_std:.5f}, FAR: {far:.5f}"
    )

    # --- ROC Curve Plot ---
    os.makedirs("eval/eval_output", exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC - {dataset_name}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC for {dataset_name}")
    plt.legend(loc="lower right")
    save_path = os.path.join("eval/eval_output", f"{dataset_name}_roc_curve.png")
    plt.savefig(save_path)
    plt.close()

    # ================================
    # 1) Pick a single best threshold from fold with highest test accuracy
    # ================================
    best_fold_idx = np.argmax(accuracy_folds)
    chosen_thresh_idx = best_threshold_indices[best_fold_idx]
    best_threshold = thresholds[chosen_thresh_idx]

    # 2) Compute predicted same/diff at that threshold
    predict_issame = dist < best_threshold
    actual_issame_arr = np.array(issame_list, dtype=bool)

    # 3) Indices for same pairs
    same_mask = actual_issame_arr
    diff_mask = ~actual_issame_arr

    # -- same accuracy
    correct_same = np.sum(np.logical_and(predict_issame, same_mask))
    total_same = np.sum(same_mask)
    same_acc = correct_same / (total_same + 1e-12)

    # -- different accuracy
    correct_diff = np.sum(np.logical_and(~predict_issame, diff_mask))
    total_diff = np.sum(diff_mask)
    diff_acc = correct_diff / (total_diff + 1e-12)

    # 4) Log them
    logging.info(
        f"[{dataset_name}] Threshold: {best_threshold:.3f} => "
        f"Same Acc: {same_acc:.4f}, Diff Acc: {diff_acc:.4f}"
    )

    # ================================
    # Plot one same pair & one not-same pair
    # ================================
    # Make subdir for this dataset
    dataset_dir = os.path.join("eval/eval_output", dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # A) Find one same pair
    same_idx = None
    for i, is_same in enumerate(issame_list):
        if is_same:
            same_idx = i
            break

    # B) Find one different pair
    diff_idx = None
    for i, is_same in enumerate(issame_list):
        if not is_same:
            diff_idx = i
            break

    def plot_pair(idx, filename):
        """
        Plots the images for pair at 'idx' (indices1[idx], indices2[idx]).
        Saves to 'filename'.
        """
        if idx is None:
            return

        idxA = indices1[idx]
        idxB = indices2[idx]

        # Load the images from the dataset
        imgA, _ = dataset[idxA]
        imgB, _ = dataset[idxB]
        # Make them [H,W,C] for imshow
        imgA = imgA.permute(1, 2, 0).cpu().numpy()
        imgB = imgB.permute(1, 2, 0).cpu().numpy()

        # Plot side-by-side
        plt.figure(figsize=(6,3))
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(imgA)
        ax1.axis("off")
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(imgB)
        ax2.axis("off")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # Plot same pair => "same.png"
    plot_pair(same_idx, os.path.join(dataset_dir, "same.png"))
    # Plot not-same pair => "notsame.png"
    plot_pair(diff_idx, os.path.join(dataset_dir, "notsame.png"))
    # ================================

    # PCA
    if visualize_pca:
        # Step A: pick 15 same + 15 different pairs (if available)
        same_indices = np.where(issame_list == True)[0]
        diff_indices = np.where(issame_list == False)[0]

        np.random.shuffle(same_indices)
        np.random.shuffle(diff_indices)

        n_same = min(15, len(same_indices))
        n_diff = min(15, len(diff_indices))
        chosen_same = same_indices[:n_same]
        chosen_diff = diff_indices[:n_diff]
        chosen = np.concatenate([chosen_same, chosen_diff], axis=0)

        if chosen.size < 2:
            logging.info("Not enough pairs to visualize PCA.")
        else:
            # Step B: gather embeddings for *both* images in these chosen pairs
            # shape => (2*N, embed_dim)
            emb_stack = []
            pair_to_indices = []  # track which pair we belong to
            for idx in chosen:
                emb_stack.append(embeddings1[idx])  # first image
                emb_stack.append(embeddings2[idx])  # second image
                pair_to_indices.append(idx)  # so we know the pair index
                pair_to_indices.append(idx)  # repeated for second image

            emb_stack = np.array(emb_stack)  # shape (2*N, emb_dim)
            # Step C: do a 2D PCA across all these embeddings
            pca = PCA(n_components=2)
            emb_2d = pca.fit_transform(emb_stack)  # shape (2*N, 2)

            # Step D: find global min_x, max_x, min_y, max_y to keep a consistent scale
            min_x, max_x = np.min(emb_2d[:,0]), np.max(emb_2d[:,0])
            min_y, max_y = np.min(emb_2d[:,1]), np.max(emb_2d[:,1])

            # Step E: also store images so we can overlay them
            # We'll map each row in emb_stack to the actual image
            images_for_plot = []
            for i, p_idx in enumerate(pair_to_indices):
                # If i is even => that row is the "first" image of pair p_idx
                # If i is odd => second image
                if (i % 2) == 0:
                    real_idx = indices1[p_idx]
                else:
                    real_idx = indices2[p_idx]

                img, _ = dataset[real_idx]
                img_np = img.permute(1,2,0).numpy()  # shape [H,W,C]
                images_for_plot.append(img_np)

            # Step F: Now for each pair in chosen, we create a separate figure with just its 2 points
            pca_folder = os.path.join("eval/eval_output", dataset_name, "pca_pairs")
            os.makedirs(pca_folder, exist_ok=True)

            for pair_i, pair_idx in enumerate(chosen):
                # The embeddings for this pair_i are in emb_2d for two rows:
                # 2*pair_i and 2*pair_i+1
                iA = 2*pair_i
                iB = 2*pair_i + 1

                xA, yA = emb_2d[iA]
                xB, yB = emb_2d[iB]
                imgA = images_for_plot[iA]
                imgB = images_for_plot[iB]

                # distance
                pair_dist = dist[pair_idx]
                # ground-truth label
                label_is_same = issame_list[pair_idx]
                label_str = "same" if label_is_same else "not-same"

                # best threshold => best_threshold
                # predicted label
                predicted_is_same = (pair_dist < best_threshold)
                pred_str = "same" if predicted_is_same else "not-same"

                # Step G: create a figure
                plt.figure(figsize=(6,6))
                ax = plt.gca()
                # Set axis range to global
                ax.set_xlim(min_x - 1, max_x + 1)
                ax.set_ylim(min_y - 1, max_y + 1)

                # Plot the two points
                colorA = "blue" if label_is_same else "red"
                colorB = colorA  # could do the same color or something else
                ax.scatter(xA, yA, c=colorA, s=40)
                ax.scatter(xB, yB, c=colorB, s=40)

                # Overlay images
                def plot_image_at(x, y, arr_img):
                    imagebox = OffsetImage(arr_img, zoom=0.3)
                    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                    ax.add_artist(ab)

                plot_image_at(xA, yA, imgA)
                plot_image_at(xB, yB, imgB)

                # Annotate text
                text_str = (
                    f"Dist={pair_dist:.4f}\n"
                    f"Label={label_str}\n"
                    f"Pred={pred_str}"
                )
                ax.text(0.05, 0.95, text_str,
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

                ax.set_title(f"PCA Pair {pair_idx} ({label_str} vs. {pred_str})")
                out_png = os.path.join(pca_folder, f"pca_pair_{pair_idx}.png")
                plt.savefig(out_png)
                plt.close()

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
    num_pos_pairs=3000,
    num_neg_pairs=3000,
    batch_size=64,
    nrof_folds=10,
    threshold_start=0,
    threshold_end=4,
    threshold_step=0.01,
    visualize_pca=False,
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
    age30_dataset = time_load_dataset(root_datasets.AGEDB_30_ROOT, transform_pipeline, 0)
    ca_lfw_dataset = time_load_dataset(root_datasets.CA_LFW_ROOT, transform_pipeline, 0)
    cp_lfw_dataset = time_load_dataset(root_datasets.CP_LFW_ROOT, transform_pipeline, 0)
    ijbb_dataset = time_load_dataset(root_datasets.IJBB_ROOT, transform_pipeline, 0)
    ijbc_dataset = time_load_dataset(root_datasets.IJBC_ROOT, transform_pipeline, 0)
    lfw_dataset = time_load_dataset(root_datasets.LFW_ROOT, transform_pipeline, 0)

    eval_datasets = {
        "AgeDB30": age30_dataset,
        "CALFW": ca_lfw_dataset,
        "CPLFW": cp_lfw_dataset,
        "IJB-B": ijbb_dataset,
        "IJB-C": ijbc_dataset,
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
            output_size=10,  # Doesn't matter; only using feature_extractor
        )
        feature_extractor = model.features
    elif network == "camilenet130k":
        logging.info("Using CamileNet130k model")
        model = CamileNet130k(
            input_channels=3,
            hidden_size=embedding_size,
            embedding_size=embedding_size,
            output_size=10,
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
    for name, ds in eval_datasets.items():
        results = evaluate_dataset(
            ds,
            feature_extractor,
            device,
            dataset_name=name,
            num_pos_pairs=num_pos_pairs,
            num_neg_pairs=num_neg_pairs,
            batch_size=batch_size,
            nrof_folds=nrof_folds,
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            threshold_step=threshold_step,
            visualize_pca=visualize_pca,
        )
        metrics_data[name] = results

    # Log overall accuracy for each dataset
    for name, res in metrics_data.items():
        logging.info(f"{name} -> Overall Accuracy: {res['acc_mean']:.4f} ± {res['acc_std']:.4f}")


if __name__ == "__main__":
    options = parse_args()
    logging.info(f"Configuration: {options}")
    main(
        use_cuda=options.use_cuda,
        seed=options.seed,
        network=options.network,
        embedding_size=options.embedding_size,
        checkpoint_str=options.checkpoint_str,
        num_pos_pairs=options.num_pos_pairs,
        num_neg_pairs=options.num_neg_pairs,
        batch_size=options.batch_size,
        nrof_folds=options.nrof_folds,
        threshold_start=options.threshold_start,
        threshold_end=options.threshold_end,
        threshold_step=options.threshold_step,
        visualize_pca=options.visualize_pca,
    )

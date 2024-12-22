import logging
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 1) Create a session string from the current date/time
session_time = time.strftime("%Y-%m-%d_%H-%M-%S")

# 2) Use that session string in the log file name
log_filename = f"eval_{session_time}.log"

# 3) Configure logging (including our session info in the format)
logging.basicConfig(
    filename=log_filename,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
from models import CamileNet

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
    num_pos_pairs: int = 3000
    num_neg_pairs: int = 3000
    batch_size: int = 64
    nrof_folds: int = 10

def get_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for MAML-based few-shot learning training.
    """
    parser = argparse.ArgumentParser(description='MAML-based few-shot learning')
    parser.add_argument('--use-cuda', type=int, default=1, help='Use CUDA (1) or CPU (0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument("--network", type=str, default="edgeface_xs_gamma_06", help="Network architecture to use - camilenet | edgeface_xs_gamma_06")
    parser.add_argument("--type_network", type=str, default="edgeface_xs_gamma_06", help="Type of network architecture to use")
    parser.add_argument("--embedding_size", type=int, default=512, help="Size of the embedding layer")
    parser.add_argument("--num_pos_pairs", type=int, default=3000, help="Number of positive pairs")
    parser.add_argument("--num_neg_pairs", type=int, default=3000, help="Number of negative pairs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--nrof_folds", type=int, default=10, help="Number of folds for evaluation")

    parser.add_argument("--checkpoint_str", type=str, default="checkpoint/checkpoint.pth", help="Path to the checkpoint file")
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
        nrof_folds=args.nrof_folds
    )


def time_load_dataset(root_dir, transform_pipeline, min_samples_per_identity):
    time_start = time.time()
    dataset = IdentityImageDataset(
        root_dir=root_dir,
        transform=transform_pipeline,
        min_samples_per_identity=min_samples_per_identity
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
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                               np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    """Compute TPR, FPR, and accuracy (via KFold cross-validation)."""
    assert (embeddings1.shape[0] == embeddings2.shape[0])
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
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set]
            )

        # Finally, record accuracy at best threshold for this fold
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set]
        )

    # Average TPR/FPR across folds
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    """Computes the validation rate (VAL) at given FAR target."""
    assert (embeddings1.shape[0] == embeddings2.shape[0])
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
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
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
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame))
    )
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    """Full evaluation: TPR/FPR/Accuracy across thresholds + VAL/FAR."""
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    tpr, fpr, accuracy = calculate_roc(
        thresholds,
        embeddings1,
        embeddings2,
        np.asarray(actual_issame),
        nrof_folds=nrof_folds,
        pca=pca
    )
    # For val and far
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(
        thresholds,
        embeddings1,
        embeddings2,
        np.asarray(actual_issame),
        far_target=1e-3,
        nrof_folds=nrof_folds
    )
    return tpr, fpr, accuracy, val, val_std, far

def build_pairs_from_dataset(
    dataset, 
    num_pos_pairs=3000, 
    num_neg_pairs=3000
):
    """
    dataset[i] should return (image, label).
    We group indices by label, then sample 'num_pos_pairs' same pairs
    and 'num_neg_pairs' different pairs.

    Returns:
        indices1, indices2: lists of dataset indices for each pair
        issame_list: list of booleans (True if same identity, False otherwise)
    """
    # 1) Group all dataset indices by identity label
    label_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        _, label = dataset[i]
        label_to_indices[label].append(i)

    label_groups = list(label_to_indices.items())  
    # label_groups is like [(labelA, [idx1, idx2, ...]), (labelB, [...]), ...]

    # 2) Build "same" (positive) pairs
    same_pairs = []
    for label, idxs in label_groups:
        if len(idxs) < 2:
            continue
        random.shuffle(idxs)
        # For demonstration, form consecutive pairs
        # Alternatively, you could randomly choose pairs from idxs
        for i in range(len(idxs) - 1):
            same_pairs.append((idxs[i], idxs[i+1], True))
    # Limit number of positive pairs
    same_pairs = random.sample(same_pairs, min(len(same_pairs), num_pos_pairs))

    # 3) Build "different" (negative) pairs
    diff_pairs = []
    count_diffs = 0
    while count_diffs < num_neg_pairs:
        # pick a random label group
        la, idxsA = random.choice(label_groups)
        idxA = random.choice(idxsA)
        # pick a different label group
        lb, idxsB = random.choice(label_groups)
        if lb == la:
            continue  # same label => skip; we need a different label
        idxB = random.choice(idxsB)
        diff_pairs.append((idxA, idxB, False))
        count_diffs += 1

    # 4) Combine and shuffle
    all_pairs = same_pairs + diff_pairs
    random.shuffle(all_pairs)

    # 5) Unpack into separate arrays
    indices1 = [p[0] for p in all_pairs]
    indices2 = [p[1] for p in all_pairs]
    issame_list = [p[2] for p in all_pairs]

    return indices1, indices2, issame_list

def compute_pairwise_embeddings(
    model,
    dataset,
    device,
    indices1,
    indices2,
    batch_size=64
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


def evaluate_dataset(
    dataset,
    model,
    device,
    dataset_name="Unknown",
    num_pos_pairs=3000,
    num_neg_pairs=3000,
    batch_size=64,
    nrof_folds=10
):
    logging.info(f"Evaluating dataset: {dataset_name}")

    # Step 1: Build pairs with separate numbers of pos/neg
    indices1, indices2, issame_list = build_pairs_from_dataset(
        dataset, 
        num_pos_pairs=num_pos_pairs, 
        num_neg_pairs=num_neg_pairs
    )

    # Step 2: Compute embeddings (same code as before)
    embeddings = compute_pairwise_embeddings(
        model, dataset, device, indices1, indices2, batch_size=batch_size
    )

    # Step 3: Evaluate with the typical TPR/FPR/Accuracy logic
    tpr, fpr, accuracy, val, val_std, far = evaluate(
        embeddings,
        issame_list,
        nrof_folds=nrof_folds,
        pca=0
    )
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)

    logging.info(
        f"[{dataset_name}] Accuracy: {acc_mean:.5f} ± {acc_std:.5f}, "
        f"VAL: {val:.5f} ± {val_std:.5f}, FAR: {far:.5f}"
    )

    return {
        "tpr": tpr,
        "fpr": fpr,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "val": val,
        "val_std": val_std,
        "far": far
    }


def main(
    use_cuda=1,
    seed=42,
    network='edgeface_xs_gamma_06',
    embedding_size=512,
    checkpoint_str="checkpoint/checkpoint.pth",
    num_pos_pairs=3000,
    num_neg_pairs=3000,
    batch_size=64,
    nrof_folds=10
):
    logging.info("Starting eval script")
    use_cuda = bool(use_cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU device")

    # Create a face detection + alignment transform
    face_detect_align = FaceDetectAlign(
        detector=None,  # Let it auto-create MTCNN if installed
        output_size=(112, 112),
        box_enlarge=1.3
    )
    transform_pipeline = transforms.Compose([
        face_detect_align,
        transforms.ToTensor()
    ])

    # Load datasets
    age30_dataset = time_load_dataset(root_datasets.AGEDB_30_ROOT, transform_pipeline, 0)
    ca_lfw_dataset = time_load_dataset(root_datasets.CA_LFW_ROOT, transform_pipeline, 0)
    cfp_fp_dataset = time_load_dataset(root_datasets.CFP_FP_ROOT, transform_pipeline, 0)
    cp_lfw_dataset = time_load_dataset(root_datasets.CP_LFW_ROOT, transform_pipeline, 0)
    ijbb_dataset = time_load_dataset(root_datasets.IJBB_ROOT, transform_pipeline, 0)
    ijbc_dataset = time_load_dataset(root_datasets.IJBC_ROOT, transform_pipeline, 0)
    lfw_dataset = time_load_dataset(root_datasets.LFW_ROOT, transform_pipeline, 0)

    eval_datasets = {
        "AgeDB30": age30_dataset,
        "CALFW": ca_lfw_dataset,
        "CFP-FP": cfp_fp_dataset,
        "CPLFW": cp_lfw_dataset,
        "IJB-B": ijbb_dataset,
        "IJB-C": ijbc_dataset,
        "LFW": lfw_dataset
    }

    # Initialize model
    if network == "edgeface_xs_gamma_06":
        feature_extractor = get_model(
            network, dropout=0.0, fp16=False, num_features=embedding_size
        )
    elif network == "camilenet":
        model = CamileNet(
            input_channels=3,
            hidden_size=64,
            embedding_size=embedding_size,
            output_size=10 # Doesn't matter, will get only feature extractor
        )
        feature_extractor = model.features

    checkpoint = torch.load(checkpoint_str, map_location=device)
    feature_extractor.load_state_dict(checkpoint['feature_extractor'])
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
            nrof_folds=nrof_folds
        )
        metrics_data[name] = results

    # accuracy per dataset:
    for name, res in metrics_data.items():
        logging.info(
            f"{name} -> Accuracy: {res['acc_mean']:.4f} ± {res['acc_std']:.4f}"
        )


# Accuracy: LFW, CPLFW, CALFW, CFP-FP, and AgeDB30
# True Accept Rate: IJB-C dataset, where they applied the true acceptance rate (TAR) at a false acceptance rate (FAR) of 10−4, denoted as TAR at FAR=10−4
if __name__ == '__main__':
    options = parse_args()
    main(
        use_cuda=options.use_cuda,
        seed=options.seed,
        network=options.network,
        embedding_size=options.embedding_size,
        checkpoint_str=options.checkpoint_str,
        num_pos_pairs=options.num_pos_pairs,
        num_neg_pairs=options.num_neg_pairs,
        batch_size=options.batch_size,
        nrof_folds=options.nrof_folds
    )

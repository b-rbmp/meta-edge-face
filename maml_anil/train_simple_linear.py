import logging
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

import numpy as np
import random
import learn2learn as l2l
from learn2learn.data.transforms import (
    FusedNWaysKShots,
    LoadData,
    RemapLabels,
    ConsecutiveLabels,
)
from dataset.face_identity_dataset import FaceDetectAlign
from torchvision import transforms

from dataset import root_datasets
from models import get_model
from utils import time_load_dataset, time_load_meta_dataset, time_load_folded_dataset
from maml_anil.config import parse_args
import wandb

# Create a face detection + alignment transform
face_detect_align = FaceDetectAlign(
    detector=None,  # Let it auto-create MTCNN if installed
    output_size=(112, 112),
    box_enlarge=1.3,  # Enlarge bounding box slightly
)

# Compose with other transforms, e.g. ToTensor
transform_pipeline = transforms.Compose([face_detect_align, transforms.ToTensor()])


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(
    batch,
    learner,
    feature_extractor,
    loss_fn,
    adaptation_steps,
    shots,
    ways,
    device=None,
    max_batch_size=None,
    allow_nograd=None,
):

    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    # If max batch size is not None, separate the data into batches pass through the feature extractor in a for loop and then recombine
    if max_batch_size is not None:
        data = data.view(-1, max_batch_size, *data.shape[1:])
        labels = labels.view(-1, max_batch_size)
        data = [feature_extractor(data_batch) for data_batch in data]
        data = torch.cat(data, dim=0)
        labels = labels.view(-1)
    else:
        data = feature_extractor(data)

    # Split into adaptation/evaluation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)

    adaptation_data, adaptation_labels = (
        data[adaptation_indices],
        labels[adaptation_indices],
    )
    evaluation_data, evaluation_labels = (
        data[evaluation_indices],
        labels[evaluation_indices],
    )

    for _ in range(adaptation_steps):
        logits = learner(adaptation_data)
        train_error = loss_fn(logits, adaptation_labels)
        learner.adapt(train_error, allow_nograd=allow_nograd)

    if allow_nograd is None:
        logits = learner(evaluation_data)
        valid_error = loss_fn(logits, evaluation_labels)
        valid_accuracy = accuracy(logits, evaluation_labels)

    if allow_nograd:
        with torch.no_grad():
            logits = learner(evaluation_data)
            valid_error = loss_fn(logits, evaluation_labels)
            valid_accuracy = accuracy(logits, evaluation_labels)

    return valid_error, valid_accuracy


def main(
    ways=5,
    shots=5,
    meta_learning_rate=0.001,
    fast_learning_rate=0.1,
    adaptation_steps=5,
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
    network="edgeface_xs_gamma_06",
    embedding_size=512,
    loss_s=64.0,
    loss_m1=1.0,
    loss_m2=0.0,
    loss_m3=0.4,
    interclass_filtering_threshold=0.0,
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

    logging.info("Starting training script - Simple Linear")
    config = {
        "meta_learning_rate": meta_learning_rate,
        "fast_learning_rate": fast_learning_rate,
        "adaptation_steps": adaptation_steps,
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
        "loss_s": loss_s,
        "loss_m1": loss_m1,
        "loss_m2": loss_m2,
        "loss_m3": loss_m3,
        "interclass_filtering_threshold": interclass_filtering_threshold,
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
        2 * shots,
        logging=logging
    )
    age30_dataset = time_load_dataset(
        root_datasets.AGEDB_30_ROOT,
        transform_pipeline,
        2 * shots,
        logging=logging
    )
    bupt_dataset = time_load_dataset(
        root_datasets.BUPT_CBFACE_ROOT,
        transform_pipeline,
        2 * shots,
        logging=logging
    )
    ca_lfw_dataset = time_load_dataset(
        root_datasets.CA_LFW_ROOT,
        transform_pipeline,
        2 * shots,
        logging=logging
    ) 
    # cfp_fp_dataset = time_load_dataset(
    #     root_datasets.CFP_FP_ROOT,
    #     transform_pipeline,
    #     2 * shots,
    #     logging=logging 
    # ) # CURRENTLY BROKEN DUE TO FILESTRUCTURE HAVING TWO FOLDERS - FIX IT
    cp_lfw_dataset = time_load_dataset(
        root_datasets.CP_LFW_ROOT,
        transform_pipeline,
        2 * shots,
        logging=logging
    ) 
    ijbb_dataset = time_load_dataset(
        root_datasets.IJBB_ROOT,
        transform_pipeline,
        2 * shots,
        logging=logging
    ) 
    ijbc_dataset = time_load_dataset(
        root_datasets.IJBC_ROOT,
        transform_pipeline,
        2 * shots,
        logging=logging
    )
    lfw_dataset = time_load_dataset(
        root_datasets.LFW_ROOT,
        transform_pipeline,
        2 * shots,
        logging=logging
    ) 
    ms1mv2_datasets = time_load_folded_dataset(
        root_datasets.MS1MV2_ROOT,
        transform_pipeline,
        2 * shots,
        logging=logging
    )
    umdfaces_dataset = time_load_dataset(
        root_datasets.UMDFACES_ROOT,
        transform_pipeline,
        2 * shots,
        logging=logging
    )
    glint360_datasets = time_load_folded_dataset(
        root_datasets.GLINT360K_ROOT,
        transform_pipeline,
        2 * shots,
        logging=logging
    )

    # Load meta-datasets
    casiawebface_metadataset = time_load_meta_dataset(casiawebface_dataset, logging=logging)
    age30_metadataset = time_load_meta_dataset(age30_dataset, logging=logging)
    bupt_metadataset = time_load_meta_dataset(bupt_dataset, logging=logging)
    ca_lfw_metadataset = time_load_meta_dataset(ca_lfw_dataset, logging=logging)
    #cfp_fp_metadataset = time_load_meta_dataset(cfp_fp_dataset, logging=logging)  # CURRENTLY BROKEN DUE TO FILESTRUCTURE HAVING TWO FOLDERS - FIX IT
    cp_lfw_metadataset = time_load_meta_dataset(cp_lfw_dataset, logging=logging)
    ijbb_metadataset = time_load_meta_dataset(ijbb_dataset, logging=logging)
    ijbc_metadataset = time_load_meta_dataset(ijbc_dataset, logging=logging)
    lfw_metadataset = time_load_meta_dataset(lfw_dataset, logging=logging)
    ms1mv2_metadatasets = [time_load_meta_dataset(ms1mv2_dataset, logging=logging) for ms1mv2_dataset in ms1mv2_datasets]
    umdfaces_metadataset = time_load_meta_dataset(umdfaces_dataset, logging=logging)
    glint360_metadatasets = [time_load_meta_dataset(glint360_dataset, logging=logging) for glint360_dataset in glint360_datasets]

    # Create list of datasets to be used
    train_datasets = [casiawebface_metadataset, bupt_metadataset, umdfaces_metadataset]
    train_datasets.extend(ms1mv2_metadatasets)
    train_datasets.extend(glint360_metadatasets)
    valid_datasets = [age30_metadataset, ca_lfw_metadataset, cp_lfw_metadataset, ijbb_metadataset, ijbc_metadataset, lfw_metadataset] # Missing CFP-FP due to broken dataset
    
    train_tasksets = []
    train_tasksets_identity_size = []
    for dataset in train_datasets:
        identity_size = len(dataset)
        logging.info(f"Number of samples in {dataset}: {identity_size}")
        train_tasksets_identity_size.append(identity_size)
        train_taskset = l2l.data.TaskDataset(
            dataset,
            task_transforms=[
                FusedNWaysKShots(dataset, n=ways, k=2 * shots),
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
                FusedNWaysKShots(dataset, n=ways, k=2 * shots),
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

    # margin_loss = CombinedMarginLoss( see notes below
    #     loss_s,
    #     loss_m1,
    #     loss_m2,
    #     loss_m3,
    #     interclass_filtering_threshold
    # )

    feature_extractor = get_model(
        network, dropout=0.0, fp16=False, num_features=embedding_size
    )

    # This is made for cases where you have a huge number of classes as it keeps the weights of the classes, not compatible with MAML
    # head = PartialFC_SingleGPU(
    #     margin_loss=margin_loss,
    #     embedding_size=embedding_size,
    #     num_classes=ways,
    #     fp16=False,
    # )
    # Also doesn't seem to work
    # head = NormalizedLinearHeadWithCombinedMargin(
    #     embedding_size=embedding_size,
    #     num_classes=ways,
    #     margin_loss=margin_loss
    # )

    # Create simple linear head
    head = nn.Linear(embedding_size, ways, bias=True)
    torch.nn.init.xavier_uniform_(head.weight.data, gain=1.0)
    torch.nn.init.constant_(head.bias.data, 0.0)

    feature_extractor.to(device)
    head = l2l.algorithms.MAML(head, lr=fast_learning_rate)
    head.to(device)

    all_parameters = list(feature_extractor.parameters()) + list(head.parameters())
    num_params = sum([np.prod(p.size()) for p in all_parameters])
    logging.info(f"Total number of parameters: {num_params / 1e6:.2f}M")

    optimizer = torch.optim.Adam(all_parameters, lr=meta_learning_rate)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    # Make sure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if resume_from_checkpoint:
        checkpoint = torch.load(save_path)
        resume_epoch = checkpoint["epoch"]
        feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logging.info(f"Resuming training from epoch {resume_epoch}")
    else:
        resume_epoch = 0

    best_meta_val_error = float("inf")
    patience_counter = 0

    iteration = resume_epoch
    iterations += resume_epoch

    # Log session start info:
    logging.info(f"Session started at: {session_time}")
    logging.info(f"Initial iteration: {iteration} (Running until {iterations - 1})")

    for iteration in range(iterations):
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0

        # Meta-train & Meta-validation steps
        for _ in range(meta_batch_size):
            # Meta-training
            learner = head.clone()
            # Sample from one of the training tasksets
            train_tasks = random.choices(train_tasksets, weights=prob_train, k=1)[0]
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                learner,
                feature_extractor,
                loss_fn,
                adaptation_steps,
                shots,
                ways,
                device,
                max_batch_size,
            )
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

        avg_train_error = meta_train_error / meta_batch_size
        avg_train_accuracy = meta_train_accuracy / meta_batch_size

        logging.info(f"\nIteration: {iteration} / {iterations}")
        logging.info(f"Meta Train Loss: {avg_train_error:.4f}")
        logging.info(f"Meta Train Accuracy: {avg_train_accuracy:.4f}")

        for p in all_parameters:
            p.grad.data.mul_(1.0 / meta_batch_size)
        torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=5.0)
        optimizer.step()

        # Evaluate on Meta-Test tasks for early stopping
        optimizer.zero_grad()
        # Free GPU memory
        torch.cuda.empty_cache()
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for _ in range(meta_batch_size):
            learner = head.clone()
            # Sample from one of the validation tasksets using the weighted probability
            valid_tasks = random.choices(valid_tasksets, weights=prob_valid, k=1)[0]
            batch = valid_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch,
                learner,
                feature_extractor,
                loss_fn,
                adaptation_steps,
                shots,
                ways,
                device,
                max_batch_size,
                allow_nograd=True,
            )
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Instead of backpropagating, we don't need to store the gradients
            # evaluation_error.backward()
            # Free the gradients
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        meta_valid_error /= meta_batch_size
        meta_valid_accuracy /= meta_batch_size

        logging.info(f"Meta Val Loss: {meta_valid_error:.4f}")
        logging.info(f"Meta Val Accuracy: {meta_valid_accuracy:.4f}")

        if use_wandb:
            wandb.log(
                {
                    "meta_train_error": avg_train_error,
                    "meta_train_accuracy": avg_train_accuracy,
                    # You can also log validation metrics
                    "meta_val_error": meta_valid_error,
                    "meta_val_accuracy": meta_valid_accuracy,
                }
            )

        # Early stopping logic
        if meta_valid_error < best_meta_val_error:
            logging.info(
                f"New best meta-val loss "
                f"({best_meta_val_error:.4f} -> {meta_valid_error:.4f}). "
                f"Saving checkpoint."
            )
            best_meta_val_error = meta_valid_error
            patience_counter = 0

            checkpoint = {
                "epoch": iteration,
                "feature_extractor": feature_extractor.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, save_path)

        else:
            patience_counter += 1
            logging.info(
                f"No improvement in meta-test loss. Patience: {patience_counter}"
            )
            if patience_counter >= patience:
                logging.info(
                    f"Early stopping triggered. No improvement for {patience} iterations."
                )
                break


if __name__ == "__main__":
    options = parse_args()
    main(
        ways=options.ways,
        shots=options.shots,
        meta_learning_rate=options.meta_learning_rate,
        fast_learning_rate=options.fast_learning_rate,
        adaptation_steps=options.adaptation_steps,
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
        loss_s=options.loss_s,
        loss_m1=options.loss_m1,
        loss_m2=options.loss_m2,
        loss_m3=options.loss_m3,
        resume_from_checkpoint=options.resume_from_checkpoint,
        run_str=options.run_str,
    )

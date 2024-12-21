import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import random
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
from models import get_model, PartialFC_SingleGPU
from losses.losses import CombinedMarginLoss
from lr_scheduler.lr_scheduler import PolynomialLRWarmup
from dataset.dataset import MXFaceDataset
from maml_anil.config import parse_args
import wandb

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def fast_adapt(batch,
               learner,
               feature_extractor,
               adaptation_steps,
               shots,
               ways,
               device=None):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    local_embeddings = feature_extractor(data)

    # Split into adaptation/evaluation sets
    adaptation_indices = np.zeros(local_embeddings.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)

    adaptation_data, adaptation_labels = local_embeddings[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = local_embeddings[evaluation_indices], labels[evaluation_indices]

    for _ in range(adaptation_steps):
        train_error, _ = learner(adaptation_data, adaptation_labels)
        learner.adapt(train_error)

    valid_error, predictions = learner(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

def main(
    ways=5,
    shots=5,
    meta_learning_rate=0.001,
    fast_learning_rate=0.1,
    adaptation_steps=5,
    meta_batch_size=32,
    iterations=1000,
    use_cuda=1,
    seed=42,
    number_train_tasks=20000,
    number_valid_tasks=600,
    number_test_tasks=600,
    patience=10,  # Number of iterations to wait for improvement
    save_path='checkpoint/checkpoint.pth',
    debug_mode=False,
    use_wandb=False,
    network='edgeface_xs_gamma_06',
    embedding_size=512,
    loss_s=64.0,
    loss_m1=1.0,
    loss_m2=0.0,
    loss_m3=0.4,
    interclass_filtering_threshold=0.0,
    resume_from_checkpoint=False
):
    
    if use_wandb:
        wandb.init(
            project="edgeface-maml-anil",
            entity="benchmark_bros",
            config={
                "meta_learning_rate": meta_learning_rate,
                "fast_learning_rate": fast_learning_rate,
                "adaptation_steps": adaptation_steps,
                "meta_batch_size": meta_batch_size,
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
                "resume_from_checkpoint": resume_from_checkpoint
            },
        )

    use_cuda = bool(use_cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    # Load datasets --> FORCE NEW SPLIT LATER (TODO)
    vgg_face_2_train_dataset = MXFaceDataset(root_dir="processed_data/vggface2/", train_val_test='train')
    vgg_face_2_valid_dataset = MXFaceDataset(root_dir="processed_data/vggface2/", train_val_test='val')
    

    # Load meta-datasets
    vgg_face_2_meta_train_dataset = l2l.data.MetaDataset(vgg_face_2_train_dataset)
    vgg_face_2_meta_valid_dataset = l2l.data.MetaDataset(vgg_face_2_valid_dataset)


    # Create list of datasets to be used
    train_datasets = [vgg_face_2_meta_train_dataset]
    valid_datasets = [vgg_face_2_meta_valid_dataset]

    start_time = time.time()
    union_train = l2l.data.UnionMetaDataset(train_datasets)
    union_valid = l2l.data.UnionMetaDataset(valid_datasets)
    print('Time to load UNION META datasets:', time.time() - start_time)
    total_len_labels_to_indices = 0
    for dataset in train_datasets:
        total_len_labels_to_indices += len(dataset.labels_to_indices)
    if len(union_train.labels_to_indices) == total_len_labels_to_indices:
        print('Union dataset is working properly')
    else:
        raise ValueError('Union dataset is not working properly')

    train_transforms = [
        FusedNWaysKShots(union_train, n=ways, k=2 * shots),
        LoadData(union_train),
        RemapLabels(union_train),
        ConsecutiveLabels(union_train),
    ]
    train_tasks = l2l.data.Taskset(
        union_train,
        task_transforms=train_transforms,
        num_tasks=number_train_tasks if not debug_mode else 50,
    )

    valid_transforms = [
        FusedNWaysKShots(union_valid, n=ways, k=2 * shots),
        LoadData(union_valid),
        ConsecutiveLabels(union_valid),
        RemapLabels(union_valid),
    ]
    valid_tasks = l2l.data.Taskset(
        union_valid,
        task_transforms=valid_transforms,
        num_tasks=number_valid_tasks if not debug_mode else 50,
    )


    margin_loss = CombinedMarginLoss(
        loss_s,
        loss_m1,
        loss_m2,
        loss_m3,
        interclass_filtering_threshold
    )

    feature_extractor = get_model(
        network, dropout=0.0, fp16=False, num_features=embedding_size
    )

    # feature_extractor.load_state_dict(torch.load("best_feature_extractor.pth", map_location=device)) #######################
    head = PartialFC_SingleGPU(
        margin_loss=margin_loss,
        embedding_size=embedding_size,
        num_classes=ways, # Doesn't matter since what we care is learning the feature extractor that will be used with cosine distance to classify during inference
        fp16=False,
    )
    feature_extractor.to(device)
    head = l2l.algorithms.MAML(head, lr=fast_learning_rate)
    head.to(device)


    all_parameters = list(feature_extractor.parameters()) + list(head.parameters())
    num_params = sum([np.prod(p.size()) for p in all_parameters])
    print('Total number of parameters:', num_params / 1e6, 'Millions')

    optimizer = torch.optim.Adam(all_parameters, lr=meta_learning_rate)
    
    # Make sure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if resume_from_checkpoint:
        checkpoint = torch.load(save_path)
        resume_epoch = checkpoint['epoch']
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        head.load_state_dict(checkpoint['head'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Resuming training from epoch {resume_epoch}")
    else:
        resume_epoch = 0

    best_meta_val_error = float('inf')
    patience_counter = 0

    iteration = resume_epoch
    iterations += resume_epoch
    for iteration in range(iterations):
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0

        # Meta-train & Meta-validation steps
        for _ in range(meta_batch_size):
            # Meta-training
            learner = head.clone()
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch, learner, feature_extractor, adaptation_steps, shots, ways, device
            )
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

        print('\nIteration:', iteration)
        print('Meta Train Error:', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy:', meta_train_accuracy / meta_batch_size)

        for p in all_parameters:
            p.grad.data.mul_(1.0 / meta_batch_size)
        optimizer.step()

        # Evaluate on Meta-Test tasks for early stopping
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for _ in range(meta_batch_size):
            # Validation Set

            learner = head.clone()
            batch = valid_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(
                batch, learner, feature_extractor, adaptation_steps, shots, ways, device
            )
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        meta_valid_error /= meta_batch_size
        meta_valid_accuracy /= meta_batch_size

        # Combine test and validation set results


        print('Meta Val Error:', meta_valid_error)
        print('Meta Val Accuracy:', meta_valid_accuracy)

        if use_wandb:
            wandb.log({
                "meta_train_error": meta_train_error / meta_batch_size,
                "meta_train_accuracy": meta_train_accuracy / meta_batch_size,
            })

        # Early stopping logic
        if meta_valid_error < best_meta_val_error:
            print(f"New best meta-val error ({best_meta_val_error} -> {meta_valid_error}). Saving feature extractor.")
            best_meta_val_error = meta_valid_error
            patience_counter = 0

            checkpoint = {
                'epoch': iteration,
                'feature_extractor': feature_extractor.state_dict(),
                'head': head.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            # Save the feature extractor’s state for later fine-tuning
            torch.save(checkpoint, save_path)
            
        else:
            patience_counter += 1
            print("No improvement in meta-test error. Patience:", patience_counter)
            if patience_counter >= patience:
                print("Early stopping triggered. No improvement in meta-test error for", patience, "iterations.")
                break

if __name__ == '__main__':
    options = parse_args()
    main(
        ways=options.ways,
        shots=options.shots,
        meta_learning_rate=options.meta_learning_rate,
        fast_learning_rate=options.fast_learning_rate,
        adaptation_steps=options.adaptation_steps,
        meta_batch_size=options.meta_batch_size,
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
        resume_from_checkpoint=options.resume_from_checkpoint
    )

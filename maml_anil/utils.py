import time
import sys
import os
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset.face_identity_dataset import IdentityImageDataset
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels

def time_load_folded_dataset(root_dir, transform_pipeline, min_samples_per_identity, logging) -> List[IdentityImageDataset]:
    time_start = time.time()
    # List all subdirectories in root_dir
    all_subdirs = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            all_subdirs.append(entry.name)
    #For each subdirectory, create a dataset
    datasets = []
    for subdir in all_subdirs:
        dataset = IdentityImageDataset(
            root_dir=os.path.join(root_dir, subdir),
            transform=transform_pipeline,
            min_samples_per_identity=min_samples_per_identity
        )
        datasets.append(dataset)

    time_end = time.time()
    logging.info(f"Time to load {len(all_subdirs)} folded datasets {root_dir}: {time_end - time_start:.2f}s")
    return datasets

def time_load_dataset(root_dir, transform_pipeline, min_samples_per_identity, logging):
    time_start = time.time()
    dataset = IdentityImageDataset(
        root_dir=root_dir,
        transform=transform_pipeline,
        min_samples_per_identity=min_samples_per_identity
    )
    time_end = time.time()
    logging.info(f"Time to load dataset {root_dir}: {time_end - time_start:.2f}s")
    return dataset

def time_load_meta_dataset(dataset, logging):
    time_start = time.time()
    metadataset = l2l.data.MetaDataset(dataset)
    time_end = time.time()
    logging.info(f"Time to load meta-dataset {dataset}: {time_end - time_start:.2f}s")
    return metadataset

def create_taskset(dataset, ways, shots, number_valid_tasks, debug_mode=False):
    """
    Helper function to create a TaskDataset from a given dataset.
    """
    return l2l.data.TaskDataset(
        dataset,
        task_transforms=[
            FusedNWaysKShots(dataset, n=ways, k=2 * shots),
            LoadData(dataset),
            RemapLabels(dataset),
            ConsecutiveLabels(dataset),
        ],
        num_tasks=number_valid_tasks if not debug_mode else 50,
    )
    
    

def group_into_union_and_individual(
    datasets,
    ways,
    shots,
    number_valid_tasks,
    threshold=2_000_000,
    debug_mode=False,
    logging=None
):
    """
    Splits a list of datasets into:
      1. All 'small' datasets (below threshold) combined into ONE UnionMetaDataset.
      2. Individual TaskDatasets for 'large' datasets (>= threshold).

    Returns:
      tasksets: A list of TaskDatasets.
      taskset_sizes: A list (same length as tasksets) with the sum of the lengths
                     of the underlying dataset(s) for that taskset.
    """
    small_datasets = []
    big_datasets = []

    # Separate datasets based on threshold
    for ds in datasets:
        length_ds = len(ds)
        if length_ds < threshold:
            small_datasets.append(ds)
            logging.info(f"Dataset {ds} is SMALL: {length_ds} < {threshold}")
        else:
            big_datasets.append(ds)
            logging.info(f"Dataset {ds} is BIG: {length_ds} >= {threshold}")

    # Containers for results
    tasksets = []
    taskset_sizes = []

    # 1) Create a single UnionMetaDataset for all small datasets (if any)
    if small_datasets:
        logging.info(
            f"Combining {len(small_datasets)} smaller datasets "
            "into one UnionMetaDataset."
        )
        union_meta = l2l.data.UnionMetaDataset(small_datasets)
        union_taskset = create_taskset(
            union_meta, ways, shots, number_valid_tasks, debug_mode
        )
        tasksets.append(union_taskset)
        # Calculate the total size of all small datasets combined
        sum_small = sum(len(ds) for ds in small_datasets)
        taskset_sizes.append(sum_small)

    # 2) Create individual TaskDatasets for each large dataset
    for ds in big_datasets:
        length_ds = len(ds)
        logging.info(f"Creating individual TaskDataset for large dataset {ds}.")
        tset = create_taskset(ds, ways, shots, number_valid_tasks, debug_mode)
        tasksets.append(tset)
        # The size is simply the single large dataset's length
        taskset_sizes.append(length_ds)

    # Return both the TaskDatasets and their combined sizes
    return tasksets, taskset_sizes
#!/usr/bin/env python3

import os
import sys
import shutil

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <root_dir> <num_folds>")
        sys.exit(1)

    root_dir = sys.argv[1]
    num_folds = int(sys.argv[2])

    # 1) Identify parent directory and create fold_extraction
    parent_dir = os.path.dirname(os.path.abspath(root_dir))
    fold_extraction_dir = os.path.join(parent_dir, "fold_extraction")

    # Create fold_extraction if not exists
    os.makedirs(fold_extraction_dir, exist_ok=True)

    # 2) Get list of subfolders in root_dir, excluding fold_* and fold_extraction
    #    We'll only consider immediate subdirectories (depth=1).
    all_subdirs = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            # Exclude fold_* or fold_extraction
            if entry.name.startswith("fold_"):
                continue
            if entry.name == "fold_extraction":
                continue
            all_subdirs.append(entry.name)

    total_folders = len(all_subdirs)
    if total_folders == 0:
        print(f"No subfolders to move in '{root_dir}'. Exiting...")
        sys.exit(0)

    print("Root directory       :", root_dir)
    print("Parent directory     :", parent_dir)
    print("Fold extraction dir  :", fold_extraction_dir)
    print("Total subfolders     :", total_folders)
    print("Number of folds      :", num_folds)

    folders_per_fold = total_folders // num_folds
    remainder = total_folders % num_folds

    print("Folders per fold     :", folders_per_fold)
    print("Remainder (extra)    :", remainder)
    print()

    # Sort so distribution is deterministic
    all_subdirs.sort()

    fold_index = 1
    folder_count_in_current_fold = 0

    # Create fold_1 directory
    current_fold_dir = os.path.join(fold_extraction_dir, f"fold_{fold_index}")
    os.makedirs(current_fold_dir, exist_ok=True)

    # 3) Distribute subfolders
    for folder_name in all_subdirs:
        # Decide how many should go in this fold
        if remainder > 0:
            target_count = folders_per_fold + 1
        else:
            target_count = folders_per_fold

        source_path = os.path.join(root_dir, folder_name)
        dest_path = os.path.join(current_fold_dir, folder_name)

        print(f"Moving '{folder_name}' -> fold_{fold_index}")
        shutil.move(source_path, dest_path)
        folder_count_in_current_fold += 1

        # If we've reached the target number for this fold
        if folder_count_in_current_fold == target_count:
            fold_index += 1
            folder_count_in_current_fold = 0
            # If we used an extra slot, reduce remainder
            if remainder > 0:
                remainder -= 1

            # Create the next fold directory if needed
            if fold_index <= num_folds:
                current_fold_dir = os.path.join(fold_extraction_dir, f"fold_{fold_index}")
                os.makedirs(current_fold_dir, exist_ok=True)

    print()
    print(f"Done! Folders have been split into {num_folds} fold(s)")
    print(f"You can find them in: {fold_extraction_dir}")

if __name__ == "__main__":
    main()

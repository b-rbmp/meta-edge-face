#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage information
usage() {
    echo "Usage: $0 <prefix> <quality> <dataset_str> <directory> [--align]"
    echo "  <prefix>: prefix for the dataset (e.g., "alldata", 'train', 'val', 'test')"
    echo "  <quality>: integer from 0 to 100"
    echo "  <dataset_str>: dataset identifier string"
    echo "  <directory>: path to the dataset directory"
    echo "  --align (optional): if present, aligns images before creating .rec files"
    exit 1
}

# --------------------------------------------------------------------
# 1. Parse and validate arguments
# --------------------------------------------------------------------
# We expect at least 4 arguments. A 5th optional one can be '--align'.
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Error: Invalid number of arguments."
    usage
fi

PREFIX="$1"
QUALITY="$2"
DATASET_STR="$3"
DIRECTORY="$4"

# Optional flag for alignment
ALIGN=false
if [ "$#" -eq 5 ]; then
    if [ "$5" == "--align" ]; then
        ALIGN=true
    else
        echo "Error: Unrecognized option '$5'"
        usage
    fi
fi


# Validate 'quality' argument (must be an integer between 0 and 100)
if ! [[ "$QUALITY" =~ ^[0-9]+$ ]]; then
    echo "Error: <quality> must be an integer."
    usage
fi

if [ "$QUALITY" -lt 0 ] || [ "$QUALITY" -gt 100 ]; then
    echo "Error: <quality> must be between 0 and 100."
    usage
fi

# Validate 'directory' argument (must be an existing directory)
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' does not exist."
    exit 1
fi

# --------------------------------------------------------------------
# 2. Conditionally align images if --align was specified
# --------------------------------------------------------------------
ALIGNED_DIR="$DIRECTORY"
if [ "$ALIGN" = true ]; then
    ALIGNED_DIR="aligned_data"  # Temporary folder for aligned images

    echo "============================================================"
    echo "[ALIGNMENT STAGE] Aligning images from '$DIRECTORY' into '$ALIGNED_DIR'..."
    echo "============================================================"

    # Ensure the aligned_data directory is empty or doesn't exist
    rm -rf "$ALIGNED_DIR"
    mkdir -p "$ALIGNED_DIR"

    # Call the Python alignment script
    python image_preprocessing/align_images.py --input_dir "$DIRECTORY" --output_dir "$ALIGNED_DIR"

    echo "Alignment completed. Proceeding with dataset preparation..."
else
    echo "============================================================"
    echo "[INFO] Skipping alignment because '--align' was not specified."
    echo "============================================================"
fi

# --------------------------------------------------------------------
# 3. Prepare the dataset using im2rec.py on the (aligned or original) images
# --------------------------------------------------------------------
TARGET_DIR="processed_data/${DATASET_STR}"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "============================================================"
echo "[IM2REC STAGE] Generating list with recursive option for '$PREFIX'..."
echo "============================================================"

python image_preprocessing/im2rec.py --list --recursive "$PREFIX" "$ALIGNED_DIR"

echo "============================================================"
echo "[IM2REC STAGE] Creating .rec file with quality $QUALITY and 16 threads for '$PREFIX'..."
echo "============================================================"

python image_preprocessing/im2rec.py --num-thread 16 --quality "$QUALITY" "$PREFIX" "$ALIGNED_DIR"

# --------------------------------------------------------------------
# 4. Move the generated .idx, .lst, and .rec files to the target directory
# --------------------------------------------------------------------
echo "============================================================"
echo "[MOVE FILES] Moving generated files to '$TARGET_DIR'..."
echo "============================================================"

mv "${PREFIX}.idx" "${TARGET_DIR}/${PREFIX}.idx"
mv "${PREFIX}.lst" "${TARGET_DIR}/${PREFIX}.lst"
mv "${PREFIX}.rec" "${TARGET_DIR}/${PREFIX}.rec"

# --------------------------------------------------------------------
# 5. Clean up the temporary aligned_data folder if --align was used
# --------------------------------------------------------------------
if [ "$ALIGN" = true ]; then
    echo "============================================================"
    echo "[CLEANUP] Removing temporary aligned images in '$ALIGNED_DIR'..."
    echo "============================================================"
    rm -rf "$ALIGNED_DIR"
fi

echo "============================================================"
echo "Processing for '$PREFIX' completed successfully."
echo "============================================================"

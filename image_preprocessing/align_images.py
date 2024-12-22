
#!/usr/bin/env python3
"""
align_faces.py

A script to align faces in images using the `get_aligned_face` function from the `face_alignment.align` module.

Usage:
    python align_faces.py --input_dir /path/to/input/image_folder --output_dir /path/to/output/aligned_faces

Arguments:
    --input_dir: Path to the input directory containing images organized in subdirectories.
    --output_dir: Path to the output directory where aligned images will be saved.

Example:
    python align_faces.py --input_dir ./image_folder --output_dir ./aligned_faces
"""
import math
import sys
import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#from face_alignment.align import get_aligned_face

import argparse
from PIL import Image
import logging
from pathlib import Path
from mtcnn import MTCNN


def setup_logging():
    """
    Sets up the logging configuration.
    Logs will be printed to the console and saved to a file named 'align_faces.log'.
    """
    logger = logging.getLogger('AlignFacesLogger')
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('align_faces.log')

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        args: Parsed arguments containing input_dir and output_dir.
    """
    parser = argparse.ArgumentParser(description='Align faces in images.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the input directory containing images.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory to save aligned images.')

    args = parser.parse_args()
    return args

class FaceDetectAlign:
    """
    A transform for face detection (with bounding box) and alignment.
    1. Uses MTCNN if available to get bounding box + keypoints.
    2. Aligns the face so that eyes are horizontally aligned.
    3. Crops the face bounding box and resizes to output_size.
    4. Falls back to a simple center-crop + resize if detection fails or MTCNN not installed.
    """

    def __init__(
        self,
        detector=None,
        output_size=(112, 112),
        fallback_transform: Optional[Callable] = None,
        box_enlarge: float = 1.2
    ):
        """
        Args:
            detector: A face detector with a detect_faces(img) method (MTCNN).
            output_size (tuple): (height, width) for the aligned face.
            fallback_transform (callable): Used when detection fails or is unavailable.
            box_enlarge (float): Factor to enlarge the bounding box around the face.
        """
        self.output_size = output_size
        self.box_enlarge = box_enlarge

        # If no detector provided, try to instantiate MTCNN
        if detector is not None:
            self.detector = detector
        else:
            if MTCNN is not None:
                self.detector = MTCNN(device="cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.detector = None  # No face detector available

        # Default fallback is a simple center-crop + resize
        if fallback_transform is not None:
            self.fallback_transform = fallback_transform
        else:
            from torchvision import transforms
            self.fallback_transform = transforms.Compose([
                transforms.CenterCrop(min(self.output_size)),
                transforms.Resize(self.output_size),
            ])

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Applies face detection + alignment to the PIL image. 
        Returns a cropped & aligned face of size self.output_size.
        Falls back to a center-crop if detection fails or no detector is present.
        """
        # If no detector, use fallback.
        if self.detector is None:
            return self.fallback_transform(img)

        # Convert PIL to numpy for MTCNN
        img_np = np.array(img)

        # Attempt detection
        try:
            results = self.detector.detect_faces(img_np)
        except Exception:
            return self.fallback_transform(img)

        if not results:
            # No faces detected
            return self.fallback_transform(img)

        # Use the first face detected or pick the largest bounding box
        face = max(results, key=lambda x: x['box'][2] * x['box'][3])  # largest area
        box = face.get('box', None)
        keypoints = face.get('keypoints', None)
        if not box or not keypoints:
            # Invalid detection result
            return self.fallback_transform(img)

        # Extract bounding box
        x, y, w, h = box
        # Possibly enlarge the bounding box
        cx, cy = x + w/2, y + h/2
        w2, h2 = w * self.box_enlarge, h * self.box_enlarge
        x = int(cx - w2/2)
        y = int(cy - h2/2)
        x2 = int(cx + w2/2)
        y2 = int(cy + h2/2)

        # Clip to image boundaries
        x, y = max(0, x), max(0, y)
        x2 = min(img.width, x2)
        y2 = min(img.height, y2)

        # Keypoints: typically { 'left_eye': (x, y), 'right_eye': (x, y), ... }
        left_eye = keypoints.get('left_eye', None)
        right_eye = keypoints.get('right_eye', None)

        if not left_eye or not right_eye:
            # If we don't have both eyes, just do a bounding box crop
            cropped = img.crop((x, y, x2, y2))
            return cropped.resize(self.output_size, Image.BILINEAR)

        # Align face by rotating so that left_eye and right_eye are horizontal
        aligned = self._align_and_crop(
            img, left_eye, right_eye, (x, y, x2, y2), self.output_size
        )
        return aligned

    @staticmethod
    def _align_and_crop(
        img: Image.Image,
        left_eye: Tuple[int, int],
        right_eye: Tuple[int, int],
        face_box: Tuple[int, int, int, int],
        output_size: Tuple[int, int]
    ) -> Image.Image:
        """
        1. Computes the angle between the eyes.
        2. Rotates the entire image around the midpoint of the eyes.
        3. Crops the bounding box region.
        4. Resizes to output_size.
        """
        (x, y, x2, y2) = face_box

        # Convert eye coords to floats
        lx, ly = float(left_eye[0]), float(left_eye[1])
        rx, ry = float(right_eye[0]), float(right_eye[1])

        # Midpoint between eyes
        eye_center_x = (lx + rx) / 2.0
        eye_center_y = (ly + ry) / 2.0

        # Angle in degrees to rotate so eyes are horizontal
        dx = rx - lx
        dy = ry - ly
        angle = math.degrees(math.atan2(dy, dx))

        # Rotate around the eye midpoint
        img_rot = FaceDetectAlign._rotate_around_point(
            img, angle, (eye_center_x, eye_center_y)
        )

        # Crop face region after rotation
        # Must recast box coords as they shift slightly if rotation is large,
        # but for mild rotations, this typically suffices.
        cropped = img_rot.crop((x, y, x2, y2))

        # Resize to the desired output
        return cropped.resize(output_size, Image.BILINEAR)

    @staticmethod
    def _rotate_around_point(
        img: Image.Image,
        angle: float,
        center: Tuple[float, float]
    ) -> Image.Image:
        """
        Rotates the entire PIL image by 'angle' degrees around 'center'.
        Returns a new rotated image.
        """
        # PIL rotation uses the upper-left corner as the origin.
        # We need to translate to center -> rotate -> translate back.
        x, y = center
        # Create an expanded canvas to prevent corners from getting cropped
        # or set expand=True below in rotate.
        # We'll do a simpler approach with expand=True to keep the whole image.
        # Then we re-crop the bounding box region later.
        # The translate for center is done by specifying the 'center' argument in rotate (Pillow>=8.0).
        return img.rotate(-angle, center=center, expand=True, resample=Image.BILINEAR)
    
def process_images(input_dir, output_dir, logger):
    """
    Processes all images in the input directory, aligns faces, and saves them to the output directory.

    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
        logger (logging.Logger): Logger for logging information and errors.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error(f"Input directory '{input_dir}' does not exist.")
        return

    if not input_path.is_dir():
        logger.error(f"Input path '{input_dir}' is not a directory.")
        return
    
    custom_get_aligned_face = FaceDetectAlign(
        detector=None,  # Let it auto-create MTCNN if installed
        output_size=(112, 112),
        box_enlarge=1.3  # Enlarge bounding box slightly
    )


    # Create the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory is set to '{output_dir}'.")

    # Supported image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # Traverse the input directory
    for root, dirs, files in os.walk(input_dir):
        # Compute the relative path from input_dir to current root
        rel_path = os.path.relpath(root, input_dir)
        # Compute the corresponding directory in the output directory
        output_subdir = output_path / rel_path
        output_subdir.mkdir(parents=True, exist_ok=True)

        for file in files:
            file_ext = Path(file).suffix.lower()
            if file_ext not in IMAGE_EXTENSIONS:
                logger.warning(f"Skipping non-image file: {file}")
                continue

            input_file_path = Path(root) / file
            output_file_path = output_subdir / file

            try:
                # Open image file as PIL image
                with Image.open(input_file_path) as img:
                    img = img.convert('RGB')

                # Get the aligned face
                #aligned_face = get_aligned_face(str(input_file_path))
                aligned_face = custom_get_aligned_face(img)
                if aligned_face is not None:
                    # Save the aligned face image
                    aligned_face.save(output_file_path)
                    logger.info(f"Aligned and saved: {output_file_path}")
                else:
                    logger.warning(f"No face detected in image: {input_file_path}. Skipping.")
            except Exception as e:
                logger.error(f"Failed to process image '{input_file_path}'. Error: {e}")

def main():
    """
    The main function that orchestrates the face alignment process.
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting face alignment process.")

    # Parse command-line arguments
    args = parse_arguments()
    input_dir = args.input_dir
    output_dir = args.output_dir

    logger.info(f"Input Directory: {input_dir}")
    logger.info(f"Output Directory: {output_dir}")

    # Process images
    process_images(input_dir, output_dir, logger)

    logger.info("Face alignment process completed.")

if __name__ == "__main__":
    main()

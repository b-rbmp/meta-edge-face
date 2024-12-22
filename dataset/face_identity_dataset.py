import os
import math
from typing import Callable, Optional, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

try:
    from mtcnn import MTCNN
except ImportError:
    MTCNN = None

# ------------------------------------------------------------------------------------
# Face detection + alignment transform
# ------------------------------------------------------------------------------------
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
                self.detector = MTCNN()
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

# ------------------------------------------------------------------------------------
# Dataset + DataLoader classes
# ------------------------------------------------------------------------------------
class IdentityImageDataset(Dataset):
    """
    Each subfolder in root_dir is an identity (label).
    Each image in that subfolder is a sample of that identity.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        min_samples_per_identity: int = 1
    ):
        """
        Args:
            root_dir (str): Path to the dataset root. 
                Each subfolder => a class (identity).
            transform (callable, optional): Transform to apply to each image
                (e.g., FaceDetectAlign, augmentations, ToTensor, etc.).
            min_samples_per_identity (int): Skip identities with fewer samples.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.min_samples_per_identity = min_samples_per_identity
        # Subfolder name => label_id
        self.class_to_idx = self._find_class_indices(root_dir)
        self.samples = self._gather_samples(root_dir, self.class_to_idx, min_samples_per_identity)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Loads image, applies transform, returns (image, label).
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        return image, label

    @staticmethod
    def _find_class_indices(root_dir: str) -> dict:
        class_names = sorted(
            folder for folder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder))
        )
        return {cls_name: i for i, cls_name in enumerate(class_names)}

    @staticmethod
    def _gather_samples(
        root_dir: str,
        class_to_idx: dict,
        min_samples_per_identity: int
    ) -> List[Tuple[str, int]]:
        """
        Gather all samples (image paths, class indices), but only keep identities
        (subfolders) that have at least min_samples_per_identity images.
        Also prints how many identities are removed in absolute number and percentage.
        """
        # Keep track of how many identities we skip vs. total
        num_identities_total = len(class_to_idx)
        num_identities_skipped = 0

        identity_to_samples = {}
        for class_name, class_idx in class_to_idx.items():
            class_folder = os.path.join(root_dir, class_name)
            image_paths = []
            for filename in os.listdir(class_folder):
                file_path = os.path.join(class_folder, filename)
                if os.path.isfile(file_path) and IdentityImageDataset._is_image_file(filename):
                    image_paths.append(file_path)

            if len(image_paths) >= min_samples_per_identity:
                # Keep this identity
                identity_to_samples[class_idx] = image_paths
            else:
                # Skip this identity
                num_identities_skipped += 1

        # Flatten into a final list of (image_path, class_idx)
        all_samples = []
        for class_idx, image_paths in identity_to_samples.items():
            for img_path in image_paths:
                all_samples.append((img_path, class_idx))

        # Calculate and print the stats about removed identities
        if num_identities_total > 0:
            percentage_skipped = (num_identities_skipped / num_identities_total) * 100
            print(
                f"Removed {num_identities_skipped} identities "
                f"({percentage_skipped:.2f}% of total) "
                f"because they had fewer than {min_samples_per_identity} samples."
            )
        else:
            print("No identities found in the given directory.")

        return all_samples
    @staticmethod
    def _is_image_file(filename: str) -> bool:
        extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")
        return filename.lower().endswith(extensions)


def get_identity_data_loader(
    root_dir: str,
    transform: Optional[Callable] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader for an identity-based dataset:
    - Each subfolder of root_dir is a distinct identity label.
    - Transform can include face detection, alignment, and standard PyTorch transforms.

    Args:
        root_dir (str): Path to dataset root.
        transform (callable): Torch transform (e.g., FaceDetectAlign + ToTensor).
        batch_size (int): Samples per batch.
        shuffle (bool): Shuffle data after each epoch.
        num_workers (int): Worker processes for data loading.

    Returns:
        DataLoader
    """
    dataset = IdentityImageDataset(root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return loader


if __name__ == "__main__":

    from torchvision import transforms
    import matplotlib.pyplot as plt

    # Create a face detection + alignment transform
    face_detect_align = FaceDetectAlign(
        detector=None,  # Let it auto-create MTCNN if installed
        output_size=(112, 112),
        box_enlarge=1.3  # Enlarge bounding box slightly
    )

    # Compose with other transforms, e.g. ToTensor
    transform_pipeline = transforms.Compose([
        face_detect_align,
        transforms.ToTensor()
    ])

    # Create a data loader
    loader = get_identity_data_loader(
        root_dir="/home/camila/Data/CASIA-WebFace/CASIA-maxpy-clean",
        transform=transform_pipeline,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    for images, labels in loader:
        num_to_show = min(4, len(images))
        
        plt.figure(figsize=(12, 3))
        for i in range(num_to_show):
            plt.subplot(1, num_to_show, i + 1)
            img_np = images[i].permute(1, 2, 0).numpy()
            plt.imshow(img_np)
            plt.title(f"Label: {labels[i].item()}")
            plt.axis("off")
        plt.savefig('face_identity_dataset.png')
        break

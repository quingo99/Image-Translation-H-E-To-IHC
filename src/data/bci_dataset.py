import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BCIDataset(Dataset):
    """Dataset for paired HE and IHC stained tissue images (BCI dataset).

    Directory structure:
        root/{split}/HE/   - H&E stained images
        root/{split}/IHC/  - IHC (HER2) stained images
    Paired images share the same filename.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        image_size=256,
        train_crops_per_image=1,
        train_crop_mode="random",
        use_full_resolution=False,
    ):
        self.root_dir = Path(root_dir)
        self.he_dir = self.root_dir / split / "HE"
        self.image_size = int(image_size)
        self.split = split
        # How many random crops to take from each image per epoch (train only).
        # Effectively multiplies the dataset size without extra disk I/O.
        self.train_crops_per_image = int(train_crops_per_image)
        # "random": uniform random crop; "quadrant_jitter": one crop per image quadrant
        self.train_crop_mode = str(train_crop_mode).lower()
        # When True, skip center-crop and pass full-resolution images to the model
        self.use_full_resolution = bool(use_full_resolution)
        if self.train_crops_per_image < 1:
            raise ValueError("train_crops_per_image must be >= 1")
        if self.train_crop_mode not in {"random", "quadrant_jitter"}:
            raise ValueError("train_crop_mode must be one of {'random', 'quadrant_jitter'}")

        # IHC ground truth: primary location is split/IHC; for the test split
        # also check a separate groundtruth/ folder (used if labels are released later).
        if (self.root_dir / split / "IHC").is_dir():
            self.ihc_dir = self.root_dir / split / "IHC"
        elif split == "test" and (self.root_dir / "groundtruth").is_dir():
            self.ihc_dir = self.root_dir / "groundtruth"
        else:
            if split == "test":
                raise FileNotFoundError(
                    f"No IHC directory found at {self.root_dir / split / 'IHC'} "
                    f"or {self.root_dir / 'groundtruth'}"
                )
            raise FileNotFoundError(
                f"No IHC directory found at {self.root_dir / split / 'IHC'}"
            )

        # Only keep filenames that exist in BOTH directories (true paired samples).
        he_files = set(os.listdir(self.he_dir))
        ihc_files = set(os.listdir(self.ihc_dir))
        self.filenames = sorted(he_files & ihc_files)

        if len(self.filenames) == 0:
            raise RuntimeError(
                f"No paired images found in {self.he_dir} and {self.ihc_dir}."
            )

        self.transform = self._build_transforms()

    def _build_transforms(self):
        if self.split == "train":
            transforms = []
            # "random" mode: let albumentations handle the crop each call.
            # "quadrant_jitter" mode: crop is done manually in __getitem__ before this
            # transform runs, so we skip the crop step here.
            if self.train_crop_mode == "random":
                transforms.append(A.RandomCrop(self.image_size, self.image_size))
            transforms.extend(
                [
                    # Spatial augmentations applied identically to HE and IHC
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    # Normalize to [-1, 1] (matches Tanh generator output range)
                    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ToTensorV2(),
                ]
            )
            return A.Compose(
                transforms,
                # Tell albumentations to treat "ihc" as an image so spatial
                # transforms are applied consistently to both HE and IHC.
                additional_targets={"ihc": "image"},
            )
        else:
            transforms = []
            # Val/test: deterministic center crop (or full resolution if requested)
            if not self.use_full_resolution:
                transforms.append(A.CenterCrop(self.image_size, self.image_size))
            transforms.extend(
                [
                    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ToTensorV2(),
                ]
            )
            return A.Compose(
                transforms,
                additional_targets={"ihc": "image"},
            )

    def __len__(self):
        # Training length is inflated by crops_per_image so each epoch
        # sees all crops; val/test returns the actual number of images.
        if self.split == "train":
            return len(self.filenames) * self.train_crops_per_image
        return len(self.filenames)

    def _randint_inclusive(self, low, high):
        low = int(low)
        high = int(high)
        if high <= low:
            return low
        return int(np.random.randint(low, high + 1))

    def _sample_quadrant_crop(self, h, w, crop_slot):
        """Return the top-left corner (y0, x0) for a crop inside one of 4 quadrants.

        The image is split into 4 spatial quadrants (top-left, top-right,
        bottom-left, bottom-right). Each crop_slot (0-3) maps to one quadrant,
        and within that quadrant a random starting position is sampled.
        This ensures that multiple crops per image cover different image regions
        rather than clustering near the center (as purely random crops can do).
        """
        c = self.image_size
        if h < c or w < c:
            raise ValueError(
                f"image_size={c} exceeds image size ({h}, {w}) in train split."
            )

        # Quadrant index cycles 0->3 regardless of crops_per_image
        q = int(crop_slot) % 4
        y_max = h - c  # maximum valid top-left y coordinate
        x_max = w - c  # maximum valid top-left x coordinate
        half_h = h // 2
        half_w = w // 2

        # Compute the [low, high] range for the crop origin within this quadrant.
        # Clamping to [0, y_max]/[0, x_max] prevents out-of-bounds crops.
        if q == 0:  # top-left quadrant
            y_low, y_high = 0, min(max(0, half_h - c), y_max)
            x_low, x_high = 0, min(max(0, half_w - c), x_max)
        elif q == 1:  # top-right quadrant
            y_low, y_high = 0, min(max(0, half_h - c), y_max)
            x_low, x_high = min(max(0, half_w), x_max), x_max
        elif q == 2:  # bottom-left quadrant
            y_low, y_high = min(max(0, half_h), y_max), y_max
            x_low, x_high = 0, min(max(0, half_w - c), x_max)
        else:  # bottom-right quadrant
            y_low, y_high = min(max(0, half_h), y_max), y_max
            x_low, x_high = min(max(0, half_w), x_max), x_max

        y0 = self._randint_inclusive(y_low, y_high)
        x0 = self._randint_inclusive(x_low, x_high)
        return y0, x0

    def __getitem__(self, idx):
        # For training with multiple crops: idx encodes both which image and
        # which crop slot within that image.
        # e.g. with 4 crops/image: idx=5 -> image 1, crop_slot 1
        crop_slot = 0
        if self.split == "train":
            crop_slot = idx % self.train_crops_per_image
            idx = idx // self.train_crops_per_image
        fname = self.filenames[idx]

        # Load HE and IHC images; OpenCV reads as BGR so convert to RGB
        he_img = cv2.imread(str(self.he_dir / fname))
        he_img = cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB)

        ihc_img = cv2.imread(str(self.ihc_dir / fname))
        ihc_img = cv2.cvtColor(ihc_img, cv2.COLOR_BGR2RGB)

        # quadrant_jitter: manually crop HE+IHC to the same region before
        # passing to albumentations (which handles flips/rotations/normalization).
        if self.split == "train" and self.train_crop_mode == "quadrant_jitter":
            y0, x0 = self._sample_quadrant_crop(he_img.shape[0], he_img.shape[1], crop_slot)
            c = self.image_size
            he_img = he_img[y0 : y0 + c, x0 : x0 + c]
            ihc_img = ihc_img[y0 : y0 + c, x0 : x0 + c]

        # Apply the shared transform (crop + augmentation + normalize + to tensor).
        # additional_targets ensures HE and IHC receive the same spatial operations.
        transformed = self.transform(image=he_img, ihc=ihc_img)
        return {
            "he": transformed["image"],
            "ihc": transformed["ihc"],
            "filename": fname,
        }

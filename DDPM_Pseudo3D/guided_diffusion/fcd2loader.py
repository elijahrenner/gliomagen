import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class FCD2Dataset(Dataset):
    """
    Dataset for FCD2 pathological volumes.
    Expects directory structure:
        <root_dir>/
            images/
                sub-XXX.nii.gz
            labels/
                sub-XXX_roi.nii.gz
    Returns for training (test_flag=False): (input, label, path, slice_indices)
    Returns for sampling (test_flag=True): (input, path, slice_indices)
    Input channels are [voided_image, mask]; label is original image.
    """
    def __init__(self, root_dir, test_flag=False):
        super().__init__()
        self.root_dir = os.path.expanduser(root_dir)
        self.images_dir = os.path.join(self.root_dir, 'images')
        self.labels_dir = os.path.join(self.root_dir, 'labels')
        self.test_flag = test_flag
        # List image files
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.nii.gz')])
        # Full paths
        self.image_paths = [os.path.join(self.images_dir, f) for f in self.image_files]
        self.label_paths = [os.path.join(self.labels_dir, f.replace('.nii.gz', '_roi.nii.gz')) for f in self.image_files]
        # Compute slice ranges where mask is non-zero
        self.slice_ranges = []
        for label_path in self.label_paths:
            # Load and reorient mask to canonical orientation
            mask_nii = nib.as_closest_canonical(nib.load(label_path))
            mask = mask_nii.get_fdata()
            if mask.ndim != 3:
                raise ValueError(f'Expected 3D mask at {label_path}, got shape {mask.shape}')
            # Assume slices along last axis
            idxs = [i for i in range(mask.shape[2]) if mask[..., i].sum() > 0]
            self.slice_ranges.append(idxs)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        # Load and reorient image and mask to canonical orientation
        img_nii = nib.as_closest_canonical(nib.load(image_path))
        image = img_nii.get_fdata().astype(np.float32)
        msk_nii = nib.as_closest_canonical(nib.load(label_path))
        mask = msk_nii.get_fdata().astype(np.float32)
        # Create voided image (zero out lesion region)
        voided = image.copy()
        voided[mask > 0] = 0.0
        # Intensity clipping and normalization between 0 and 1
        min_val = np.quantile(image, 0.001)
        max_val = np.quantile(image, 0.999)
        image = np.clip(image, min_val, max_val)
        voided = np.clip(voided, min_val, max_val)
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
            voided = (voided - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)
            voided = np.zeros_like(voided)
        mask = (mask > 0).astype(np.float32)
        # Stack channels: voided, mask, original
        volume = np.stack([voided, mask, image], axis=0)
        volume = torch.from_numpy(volume)
        slice_range = self.slice_ranges[idx]
        # Return inputs and label for training, or inputs only for sampling
        if not self.test_flag:
            inp = volume[:2]       # (2, H, W, D)
            label = volume[2:3]    # (1, H, W, D)
            return inp, label, image_path, slice_range
        # Sampling
        inp = volume[:2]
        return inp, image_path, slice_range
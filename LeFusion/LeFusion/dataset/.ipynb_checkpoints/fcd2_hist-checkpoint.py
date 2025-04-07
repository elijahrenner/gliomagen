import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import cv2
import torchio as tio
import matplotlib.pyplot as plt

# Preprocessing transforms: Clamp, rescale intensity then crop/pad to (48, 48, 48)
PREPROCESSING_TRANSFORMS = tio.Compose([
    tio.Clamp(out_min=0, out_max=600),
    tio.RescaleIntensity(in_min_max=(0, 600), out_min_max=(-1.0, 1.0)),
    tio.CropOrPad(target_shape=(48, 48, 48))
])

PREPROCESSING_MASK_TRANSFORMS = tio.Compose([
    tio.CropOrPad(target_shape=(48, 48, 48))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1,), flip_probability=0.5),
])

class FCD2Dataset(Dataset):
    def __init__(self, root_dir='', test_txt_dir='', augmentation=False):
        """
        Expected file structure:
          - root_dir/
              - Normal/
                  - images/       (nii.gz files)  <-- not used in training
              - Pathological/
                  - images/       (nii.gz files)
                  - labels/       (nii.gz files with '_roi' appended to basename)
        Parameters:
            root_dir (str): Parent directory containing both "Normal" and "Pathological" folders.
            test_txt_dir (str): Path to a text file listing case basenames to exclude.
            augmentation (bool): Flag to enable/disable data augmentation.
        Only pathological cases with nonzero mask sum are kept.
        """
        self.root_dir = root_dir
        self.remove_test_path = test_txt_dir
        self.file_names = self.get_file_names()
        self.augmentation = augmentation
        self.preprocessing_img = PREPROCESSING_TRANSFORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSFORMS

    def train_transform(self, image, label, p):
        # Apply a random flip transform with probability p.
        train_transforms = tio.Compose([
            tio.RandomFlip(axes=(1,), flip_probability=p),
        ])
        image = train_transforms(image)
        label = train_transforms(label)
        return image, label

    def get_file_names(self):
        # Get all .nii.gz files in any subfolder of root_dir.
        all_files = glob.glob(os.path.join(self.root_dir, '**', '*.nii.gz'), recursive=True)
        # Exclude label files (which contain '_roi' in the filename)
        image_files = [f for f in all_files if '_roi' not in os.path.basename(f)]
        
        # Apply test_txt filtering if provided.
        test_file_names = set()
        if self.remove_test_path and os.path.exists(self.remove_test_path):
            with open(self.remove_test_path, 'r') as file:
                for line in file:
                    test_file_names.add(line.strip())
        
        filtered_files = []
        for f in image_files:
            # Only use pathological cases.
            if 'Pathological' not in f:
                continue

            # Skip if file is in test list.
            if os.path.basename(f)[:-7] in test_file_names:
                continue

            # Expected label file: replace "images" with "labels" and append '_roi'
            label_path = f.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)
            label_path = label_path.replace('.nii.gz', '_roi.nii.gz')
            if not os.path.exists(label_path):
                print(f"Warning: Label file {label_path} not found for {f}. Skipping.")
                continue

            # Load label and apply preprocessing to check ROI.
            try:
                mask = tio.LabelMap(label_path)
                mask = PREPROCESSING_MASK_TRANSFORMS(mask)
            except Exception as e:
                print(f"Error loading label at {label_path}: {e}. Skipping {f}.")
                continue

            if mask.data.sum() == 0:
                print(f"Warning: Label file {label_path} has zero mask sum. Skipping {f}.")
                continue

            filtered_files.append(f)
        
        return filtered_files

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def project_to_2d(mask):
        projection = torch.max(mask, dim=0)[0]
        return projection.numpy()

    @staticmethod
    def min_enclosing_circle(projection):
        points = np.column_stack(np.where(projection > 0)).astype(np.float32)
        print(points.shape)
        (x, y), radius = cv2.minEnclosingCircle(points)
        return (int(x), int(y)), int(radius)

    @staticmethod
    def create_circle_mask_2d(shape, center, radius):
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 1, thickness=-1)
        return mask

    @staticmethod
    def apply_circle_mask_to_3d(mask, circle_mask_2d):
        for i in range(mask.shape[0]):
            mask[i] = torch.from_numpy(circle_mask_2d)
        return mask

    def __getitem__(self, index):
        path = self.file_names[index]
        try:
            img = tio.ScalarImage(path)
            img = self.preprocessing_img(img)
        except Exception as e:
            raise RuntimeError(f"Error loading image at {path}: {e}")

        # For pathological cases, load corresponding label.
        label_path = path.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)
        label_path = label_path.replace('.nii.gz', '_roi.nii.gz')
        try:
            mask = tio.LabelMap(label_path)
            mask = self.preprocessing_mask(mask)
        except Exception as e:
            raise RuntimeError(f"Error loading label at {label_path}: {e}")
        # Check raw mask values and threshold if necessary
        # unique_vals = mask.data.unique()
        # print(f"Unique mask values for {label_path}: {unique_vals}")
        # if not torch.all((unique_vals == 0) | (unique_vals == 1)):
            # mask.data = (mask.data > 0).to(torch.uint8)
            # print(f"Thresholded mask values for {label_path}: {mask.data.unique()}")

        # Apply training augmentation.
        p = np.random.choice([0, 1])
        img, mask = self.train_transform(img, mask, p)
        img_data = img.data
        mask_data = mask.data

        mask_sum = mask_data.sum().float()
        if mask_sum == 0:
            # This should not occur due to filtering in get_file_names.
            raise RuntimeError(f"Unexpected zero mask sum for {path}")
        
        hist = torch.histc(img_data[mask_data > 0], bins=16, min=-1, max=1) / mask_sum

        return {
            'data': img_data,
            'label': mask_data,
            'hist': hist,
        }
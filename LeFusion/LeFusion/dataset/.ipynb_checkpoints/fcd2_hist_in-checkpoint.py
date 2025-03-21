import os
import glob
import torch
from torch.utils.data.dataset import Dataset
import torchio as tio

# Preprocessing transforms for images and masks with target shape (48, 64, 64)
PREPROCESSING_TRANSFORMS = tio.Compose([
    tio.Clamp(out_min=0, out_max=600),
    tio.RescaleIntensity(in_min_max=(0, 600), out_min_max=(-1.0, 1.0)),
    tio.Resample((3.33, 4.0, 4.0)),
    tio.CropOrPad(target_shape=(48, 64, 64))
])

PREPROCESSING_MASK_TRANSFORMS = tio.Compose([
    tio.Resample((3.33, 4.0, 4.0)),
    tio.CropOrPad(target_shape=(48, 64, 64))
])

class FCD2InDataset(Dataset):
    def __init__(self, root_dir='', test_txt_dir='', augmentation=False):
        """
        Parameters:
            root_dir (str): Root directory containing the dataset folders (Normal & Pathological).
            test_txt_dir (str): Path to the test.txt file listing cases to exclude.
            augmentation (bool): Whether to perform augmentation (not used in this inference script).
        """
        self.root_dir = root_dir
        self.remove_test_path = test_txt_dir
        self.file_names = self.get_file_names()
        self.augmentation = augmentation
        self.preprocessing_img = PREPROCESSING_TRANSFORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSFORMS

    def get_file_names(self):
        # Get all .nii.gz files recursively from the root directory
        all_file_names = glob.glob(os.path.join(self.root_dir, '**', '*.nii.gz'), recursive=True)
        # Exclude files that are labels (assumed to contain '_roi' in their filename)
        image_file_names = [f for f in all_file_names if '_roi' not in os.path.basename(f)]
        
        # Load test file names if provided to filter out cases
        test_file_names = set()
        if self.remove_test_path and os.path.exists(self.remove_test_path):
            with open(self.remove_test_path, 'r') as file:
                for line in file:
                    test_file_name = line.strip()
                    test_file_names.add(test_file_name)
                    
        # Filter out files whose basename (minus the .nii.gz extension) is in the test list.
        filtered_file_names = [
            f for f in image_file_names
            if os.path.basename(f)[:-7] not in test_file_names
        ]
        return sorted(filtered_file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]
        # Load the image and apply preprocessing
        img = tio.ScalarImage(path)
        img = self.preprocessing_img(img)

        # Determine if the image is pathological (with associated mask) or normal
        if 'Pathological' in path:
            # For pathological cases, construct the label path by replacing "images" with "labels"
            # and inserting "_roi" before the file extension.
            label_path = path.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)
            label_path = label_path.replace('.nii.gz', '_roi.nii.gz')
            mask = tio.LabelMap(label_path)
            mask = self.preprocessing_mask(mask)
        else:
            # For normal images (without a label), create a dummy zero mask.
            dummy_mask = torch.zeros_like(img.data, dtype=torch.uint8)
            mask = tio.LabelMap(tensor=dummy_mask)
            mask = self.preprocessing_mask(mask)

        # Use the image filename as GT_name
        filename = os.path.basename(path)

        return {
            'GT': img.data,
            'GT_name': filename,
            'gt_keep_mask': mask.data,
            'affine': img.affine
        }
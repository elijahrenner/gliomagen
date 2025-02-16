import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################
# Flip Function
#########################################
def flip_synthetic(slice_img):
    """
    Flip a 2D slice along the sagittal plane.
    Here we use np.flipud (vertical flip) to adjust the orientation.
    Adjust this function if your data require a different flip.
    """
    return np.flipud(slice_img)

#########################################
# Inception Feature Extractor
#########################################
class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        self.inception.fc = nn.Identity()
        self.inception.eval()
    
    def forward(self, x):
        outputs = self.inception(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs

# Instantiate the model and move to device.
inception_model = InceptionFeatureExtractor().to(device)

#########################################
# Helper Functions for Inception Features
#########################################
def preprocess_slice(slice_tensor):
    x = slice_tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # shape: (1,3,H,W)
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std

def get_inception_features(slice_tensor):
    x = preprocess_slice(slice_tensor).to(device)
    with torch.no_grad():
        features = inception_model(x)
    return features.squeeze(0).cpu().numpy()  # shape: (2048,)

#########################################
# FID Function
#########################################
def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    diff_sq = diff.dot(diff)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff_sq + np.trace(sigma1 + sigma2 - 2 * covmean)

#########################################
# KID Functions
#########################################
def calculate_kid(X, Y):
    n, d = X.shape
    m, _ = Y.shape
    KXX = (np.dot(X, X.T) / d + 1) ** 3
    KYY = (np.dot(Y, Y.T) / d + 1) ** 3
    KXY = (np.dot(X, Y.T) / d + 1) ** 3

    sum_KXX = np.sum(KXX) - np.sum(np.diag(KXX))
    sum_KYY = np.sum(KYY) - np.sum(np.diag(KYY))
    mmd = sum_KXX / (n * (n - 1)) + sum_KYY / (m * (m - 1)) - 2 * np.mean(KXY)
    return mmd

def calculate_kid_split(X, Y, num_subsets=10):
    n = X.shape[0]
    m = Y.shape[0]
    subset_size = min(n, m) // num_subsets
    kid_vals = []
    indices_X = np.random.permutation(n)
    indices_Y = np.random.permutation(m)
    for i in range(num_subsets):
        subset_X = X[indices_X[i*subset_size:(i+1)*subset_size]]
        subset_Y = Y[indices_Y[i*subset_size:(i+1)*subset_size]]
        kid_vals.append(calculate_kid(subset_X, subset_Y))
    return np.mean(kid_vals), np.std(kid_vals)

#########################################
# Utility: Extract subject ID from filename
#########################################
def extract_id(fname):
    return fname.split('_')[0]  # Extract subject ID before first underscore.

#########################################
# Utility: Load Middle Slice from a Volume
#########################################
def load_middle_slice(filepath):
    vol = nib.load(filepath).get_fdata()
    mid = vol.shape[-1] // 2
    return vol[..., mid]

#########################################
# Save Example GT-Pred Images After Flipping
#########################################
def save_example_images(gt_path, pred_path, subject_id, modality, num_examples=10, save_dir="example_images"):
    """
    Save side-by-side comparison images for 'num_examples' slices.
    GT and predicted volumes are loaded from gt_path and pred_path, respectively.
    The predicted slices are flipped using flip_synthetic().
    """
    os.makedirs(save_dir, exist_ok=True)
    gt_vol = nib.load(gt_path).get_fdata()
    pred_vol = nib.load(pred_path).get_fdata()
    total_slices = min(gt_vol.shape[-1], pred_vol.shape[-1])
    # Choose 'num_examples' equally spaced slices.
    indices = np.linspace(0, total_slices - 1, num=num_examples, dtype=int)
    
    for idx in indices:
        gt_slice = gt_vol[..., idx]
        pred_slice = pred_vol[..., idx]
        # Flip the predicted slice.
        pred_slice_flipped = flip_synthetic(pred_slice).copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(gt_slice, cmap="gray")
        axes[0].set_title(f"{modality} GT, slice {idx}")
        axes[0].axis("off")
        axes[1].imshow(pred_slice_flipped, cmap="gray")
        axes[1].set_title(f"{modality} Pred (Flipped), slice {idx}")
        axes[1].axis("off")
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{subject_id}_{modality}_slice_{idx}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {num_examples} example images for subject {subject_id}, modality {modality} in '{save_dir}'")

#########################################
# Main Evaluation Function (with flipping and saving examples)
#########################################
def evaluate():
    modalities = ['t1n', 't1c', 't2w', 't2f']
    gt_root = "val_gt"
    pred_root = "val_pred"
    results = {}

    for mod in modalities:
        print(f"\nProcessing modality: {mod}")
        gt_dir = os.path.join(gt_root, mod)
        pred_dir = os.path.join(pred_root, mod)

        # Map subject ID -> file path.
        gt_files = {extract_id(f): os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(".nii.gz")}
        pred_files = {extract_id(f): os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith(".nii.gz")}
        
        common_ids = set(gt_files.keys()).intersection(set(pred_files.keys()))
        print(f"Found {len(common_ids)} common subjects for modality {mod}")
        
        # For the first subject in common_ids, save 10 example images.
        if common_ids:
            sample_subject = list(common_ids)[0]
            save_example_images(gt_files[sample_subject], pred_files[sample_subject],
                                sample_subject, mod, num_examples=10, save_dir="example_images")
        
        ms_ssim_scores = []
        gt_features_all = []
        pred_features_all = []

        for subject_id in common_ids:
            gt_path = gt_files[subject_id]
            pred_path = pred_files[subject_id]

            gt_vol = nib.load(gt_path).get_fdata()
            pred_vol = nib.load(pred_path).get_fdata()

            num_slices = min(gt_vol.shape[-1], pred_vol.shape[-1])
            if gt_vol.shape[-1] != pred_vol.shape[-1]:
                print(f"Warning: {subject_id} ({mod}) has mismatched slices!")

            for i in range(num_slices):
                # Load ground truth slice.
                gt_slice = torch.tensor(gt_vol[..., i], dtype=torch.float32)
                # Load predicted slice, then flip (force contiguous array with .copy()).
                pred_slice = torch.tensor(pred_vol[..., i], dtype=torch.float32)
                pred_slice = torch.tensor(flip_synthetic(pred_slice.numpy()).copy(), dtype=torch.float32)
                
                # Compute MS-SSIM on 2D slices.
                gt_slice_4d = gt_slice.unsqueeze(0).unsqueeze(0)
                pred_slice_4d = pred_slice.unsqueeze(0).unsqueeze(0)
                try:
                    ssim_val = ms_ssim(gt_slice_4d, pred_slice_4d, data_range=1.0)
                    ms_ssim_scores.append(ssim_val.item())
                except Exception as e:
                    print(f"Error computing MS-SSIM for {subject_id}, slice {i}: {e}")

                # Extract Inception features.
                gt_feat = get_inception_features(gt_slice)
                pred_feat = get_inception_features(pred_slice)
                gt_features_all.append(gt_feat)
                pred_features_all.append(pred_feat)

        avg_msssim = np.mean(ms_ssim_scores) if ms_ssim_scores else float('nan')
        std_msssim = np.std(ms_ssim_scores) if ms_ssim_scores else float('nan')
        print(f"Average MS-SSIM for {mod}: {avg_msssim:.4f} ± {std_msssim:.4f}")

        gt_features_all = np.array(gt_features_all)
        pred_features_all = np.array(pred_features_all)
        
        # Compute FID.
        mu_gt = np.mean(gt_features_all, axis=0)
        sigma_gt = np.cov(gt_features_all, rowvar=False)
        mu_pred = np.mean(pred_features_all, axis=0)
        sigma_pred = np.cov(pred_features_all, rowvar=False)
        
        fid = calculate_fid(mu_gt, sigma_gt, mu_pred, sigma_pred)
        fid_std = np.std(np.linalg.norm(gt_features_all - pred_features_all, axis=1))
        print(f"FID for {mod}: {fid:.4f} ± {fid_std:.4f}")
        
        # Compute KID (with uncertainty via splitting).
        kid, kid_std = calculate_kid_split(gt_features_all, pred_features_all, num_subsets=10)
        print(f"KID for {mod}: {kid:.4f} ± {kid_std:.4f}")

        results[mod] = {
            "MS-SSIM": (avg_msssim, std_msssim),
            "FID": (fid, fid_std),
            "KID": (kid, kid_std)
        }

    return results

if __name__ == "__main__":
    final_results = evaluate()
    print("\nFinal evaluation results per modality:")
    for mod, metrics in final_results.items():
        print(f"{mod}: MS-SSIM = {metrics['MS-SSIM'][0]:.4f} ± {metrics['MS-SSIM'][1]:.4f}, "
              f"FID = {metrics['FID'][0]:.4f} ± {metrics['FID'][1]:.4f}, "
              f"KID = {metrics['KID'][0]:.4f} ± {metrics['KID'][1]:.4f}")
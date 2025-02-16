import argparse
import os
import glob
import numpy as np
import nibabel as nib
import torch
from torchvision.transforms import Compose, Lambda
from nibabel import as_closest_canonical  # for orientating scans
from diffusion_model.trainer_brats import GaussianDiffusion, num_to_groups
from diffusion_model.unet_brats import create_model
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm  # for progress bars
import multiprocessing
from functools import partial
import yaml

# Ensure matplotlib does not use any xwindows backend
plt.switch_backend('Agg')

# ----------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------

def load_verify_seg(seg_path, expected_shape):
    nii = nib.load(seg_path)  # load mask
    nii = as_closest_canonical(nii)  # reorient to canonical orientation
    data = np.array(nii.dataobj)  # load data without orientating
    if data.shape != expected_shape:
        raise ValueError(f"segmentation mask at {seg_path} has shape {data.shape}, expected {expected_shape}.")
    return data, nii.affine

def one_hot_mask(seg_3d, num_classes):
    seg_3d = seg_3d.astype(np.int64)
    out = np.zeros((num_classes, *seg_3d.shape), dtype=np.float32)
    for c in range(1, num_classes + 1):
        out[c - 1, seg_3d == c] = 1.0
    return out

def normalize(img):
    img = img - np.min(img)
    return img / np.max(img) if np.max(img) != 0 else img

# ----------------------------------------------------------------------------
# Worker Function
# ----------------------------------------------------------------------------

def process_images(worker_id, device_id, image_paths, args):
    """
    Process a subset of synthetic segmentation files on a specific GPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    num_class_labels = args.num_class_labels    # e.g. 6
    out_channels     = args.out_channels         # number of modalities (from export_modalities)
    with_condition   = args.with_condition       # e.g. True
    in_channels      = num_class_labels + out_channels

    model = create_model(
        image_size=args.input_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        in_channels=in_channels,
        out_channels=out_channels
    ).to(device)

    diffusion = GaussianDiffusion(
        denoise_fn=model,
        image_size=args.input_size,
        depth_size=args.depth_size,
        timesteps=args.timesteps,
        loss_type="l1",
        with_condition=with_condition,
        channels=out_channels
    ).to(device)

    # Only load weights if a non-empty weightfile is provided.
    if args.weightfile:
        try:
            ckpt = torch.load(args.weightfile, map_location=device)
        except TypeError:
            ckpt = torch.load(args.weightfile, map_location=device)
            import warnings
            warnings.warn(
                "Using torch.load without weights_only=True. Ensure the checkpoint is trusted.",
                FutureWarning
            )
        diffusion.load_state_dict(ckpt["ema"], strict=False)
        print(f"[Worker {worker_id}] Model weights loaded from {args.weightfile}")
    else:
        print(f"[Worker {worker_id}] No weight file provided; skipping weight loading.")

    # Condition transform: converts one-hot array to a torch tensor.
    condition_transform = Compose([
        Lambda(lambda arr: torch.tensor(arr, dtype=torch.float32)),
    ])

    # Create export directories for each modality.
    for mod in args.export_modalities:
        os.makedirs(os.path.join(args.exportfolder, mod), exist_ok=True)

    for idx, seg_path in enumerate(image_paths):
        base_name = os.path.basename(seg_path).replace(".nii.gz", "")
        print(f"[Worker {worker_id}] Processing: {seg_path} ({idx+1}/{len(image_paths)})")
        try:
            seg_data, affine = load_verify_seg(seg_path, expected_shape=(args.input_size, args.input_size, args.depth_size))
        except ValueError as e:
            print(f"[Worker {worker_id}] Error: {e}")
            continue

        seg_1hot = one_hot_mask(seg_data, num_classes=args.num_class_labels)
        seg_cond = condition_transform(seg_1hot).unsqueeze(0).to(device)

        # Generate synthetic samples based on the number specified in YAML.
        sample_batches = num_to_groups(args.num_samples, 1)
        all_samples_list = []
        for sidx, bsz in enumerate(sample_batches):
            condition_batch = seg_cond.repeat(bsz, 1, 1, 1, 1)
            with torch.no_grad():
                samples = diffusion.sample(batch_size=bsz, condition_tensors=condition_batch)
            all_samples_list.append(samples)
        all_gen = torch.cat(all_samples_list, dim=0)
        print(f"[Worker {worker_id}] Generated {all_gen.shape[0]} samples for {base_name}.")

        # Save each channel to its corresponding modality directory.
        for aug_idx in range(all_gen.shape[0]):
            sample_channels = all_gen[aug_idx].cpu().numpy()  # shape: (out_channels, H, W, D)
            for mod_idx, mod in enumerate(args.export_modalities):
                out_path = os.path.join(args.exportfolder, mod, f"{base_name}_aug_{aug_idx+1}_{mod}.nii.gz")
                nib.save(nib.Nifti1Image(sample_channels[mod_idx], affine), out_path)
        print(f"[Worker {worker_id}] Saved {all_gen.shape[0]} augmented samples for {base_name}.")

# ----------------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config.yaml", help="Path to YAML config file.")
    args_cli = parser.parse_args()

    # Load YAML configuration.
    with open(args_cli.config, "r") as f:
        config = yaml.safe_load(f)

    # Build parameters from YAML sections.
    dataset_config  = config.get("dataset", {})
    preproc_config  = config.get("preprocessing", {})
    model_config    = config.get("model", {})
    training_config = config.get("training", {})
    aug_config      = config.get("augmentation", {})

    args = argparse.Namespace()
    base_dir = "../../data"

    # Use the synthetic folder ONLY (do not combine with original masks).
    args.exportfolder = aug_config.get("output_masks_dir",
                        os.path.join(base_dir, dataset_config.get("output_root", "BraTS2024-GLI/3_gli_head_mask_normalized_multimodal").split("/")[0] + "-Synthetic"))
    # Use the synthetic folder as input.
    args.inputfolder = args.exportfolder

    # Modalities from dataset.
    args.export_modalities = dataset_config.get("modalities", ["t1n", "t1c", "t2w", "t2f"])

    # Hyperparameters.
    target_shape = preproc_config.get("target_shape", [192, 192, 144])
    args.input_size      = target_shape[0]
    args.depth_size      = target_shape[2]
    args.num_channels    = model_config.get("num_channels", 64)
    args.num_res_blocks  = model_config.get("num_res_blocks", 2)
    args.num_class_labels = model_config.get("num_class_labels", 6)
    args.out_channels    = model_config.get("out_channels", len(args.export_modalities))
    args.with_condition  = model_config.get("with_condition", True)
    args.weightfile      = model_config.get("resume_weight", "")
    args.timesteps       = training_config.get("timesteps", 1000)
    args.num_samples     = aug_config.get("n_transforms_per_mask", 1)

    print("Loaded parameters:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")

    seg_list = sorted(glob.glob(os.path.join(args.inputfolder, "*.nii.gz")))
    print(f"Total segmentation masks found: {len(seg_list)}")
    if len(seg_list) == 0:
        print("No segmentation mask files found. Exiting.")
        return

    # Distribute segmentation files evenly across available GPUs.
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPU(s).")
        gpu_ids = list(range(num_gpus))
    else:
        raise RuntimeError("No GPUs available. Please ensure at least one GPU is present.")

    distributed = [[] for _ in range(num_gpus)]
    for idx, seg_path in enumerate(seg_list):
        distributed[idx % num_gpus].append(seg_path)

    processes = []
    for worker_id, (gpu_id, image_subset) in enumerate(zip(gpu_ids, distributed)):
        if not image_subset:
            print(f"GPU {gpu_id} has no images to process. Skipping.")
            continue
        p = multiprocessing.Process(target=process_images, args=(worker_id, gpu_id, image_subset, args))
        p.start()
        processes.append(p)
        print(f"Started worker {worker_id} on GPU {gpu_id} with {len(image_subset)} images.")

    for p in processes:
        p.join()
    print("All workers have finished processing.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
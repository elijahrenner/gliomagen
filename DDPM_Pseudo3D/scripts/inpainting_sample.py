"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.

BraTS‑adapted inpainting pipeline with **overlapping sub‑batches** (stride =
window_size/2) to guarantee full pseudo‑3D context for every slice. Optional
progressive *intermediate* saves are supported to visualise how the voided
lesion region closes over diffusion timesteps.
"""

import argparse
import math
import os
import random
import sys
from datetime import datetime

import nibabel as nib
import numpy as np
try:
    from scipy.ndimage import binary_dilation
except ImportError:
    binary_dilation = None
import torch as th
import torch.distributed as dist
from prettytable import PrettyTable

# -----------------------------------------------------------------------------
# project‑local imports
# -----------------------------------------------------------------------------
sys.path.append("../")
from guided_diffusion import dist_util, logger
from guided_diffusion.fcd2loader import FCD2Dataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

# -----------------------------------------------------------------------------
# reproducibility
# -----------------------------------------------------------------------------
seed = 10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

th.backends.cudnn.enabled = True

# -----------------------------------------------------------------------------
# utility helpers (kept for completeness; unused in sampling loop itself)
# -----------------------------------------------------------------------------

def visualize(img):
    _min, _max = img.min(), img.max()
    return (img - _min) / (_max - _min)


def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2.0 * (pred * targs).sum() / (pred + targs).sum()


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# -----------------------------------------------------------------------------
# main sampling routine
# -----------------------------------------------------------------------------

def main():
    args = create_argparser().parse_args()

    # create intermediate save dir if needed
    if args.save_intermediate_interval > 0 and args.save_intermediate_dir:
        os.makedirs(args.save_intermediate_dir, exist_ok=True)

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("SAMPLING START " + str(datetime.now()))
    logger.log("args: " + str(args))

    # ---------------------------------------------------------------------
    # model & diffusion
    # ---------------------------------------------------------------------
    logger.log("creating model and diffusion …")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # ---------------------------------------------------------------------
    # dataset – one volume per batch element
    # ---------------------------------------------------------------------
    ds = FCD2Dataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,  # typically 1 volume at a time
        shuffle=False,
    )
    data_iter = iter(datal)

    total_volumes = len(ds)

    # ---------------------------------------------------------------------
    # iterate over volumes
    # ---------------------------------------------------------------------
    for vol_idx in range(total_volumes):
        print(f"Sampling file {vol_idx + 1} / {total_volumes} …")
        # batch: (B, C, H, W, D); slicedict: lesion slice indices
        batch, path, slicedict = next(data_iter)
        depth = batch.shape[-1]
        # Load image in canonical orientation to use matching affine
        orig_affine = nib.as_closest_canonical(nib.load(path[0])).affine

        # lesion slices as Python ints
        orig_slices = [int(s) for s in slicedict]

        # -----------------------------------------------------------------
        # extend lesion region by `range_padding` to give model context
        # -----------------------------------------------------------------
        if orig_slices:
            min_s, max_s = min(orig_slices), max(orig_slices)
            ext_min = max(0, min_s - args.range_padding)
            ext_max = min(depth - 1, max_s + args.range_padding)
            padded_range = list(range(ext_min, ext_max + 1))
            logger.log(
                f"Generating slices {ext_min}:{ext_max} (orig {min_s}:{max_s}, pad={args.range_padding})"
            )
        else:
            padded_range = []

        # -----------------------------------------------------------------
        # output filename
        # -----------------------------------------------------------------
        fname = os.path.basename(path[0])
        base = os.path.splitext(os.path.splitext(fname)[0])[0]
        out_file = os.path.join(args.log_dir, f"{base}_inference.nii.gz")

        if os.path.isfile(out_file):
            print("file exists already, skipping it")
            continue

        logger.log(f"Generating inference for {base} …")

        # -----------------------------------------------------------------
        # initialise generated volume with void‑channel (channel 0) of input
        # and save input voided MR + mask for alignment check
        # -----------------------------------------------------------------
        # batch shape: (1, C, H, W, D)
        generated_3D = batch[:, 0, :, :, :].squeeze().cpu().numpy()
        # Save voided input and mask volumes
        inputs_dir = os.path.join(args.log_dir, 'inputs')
        os.makedirs(inputs_dir, exist_ok=True)
        # Voided MR
        voided_path = os.path.join(inputs_dir, f"{base}_voided_input.nii.gz")
        nib.save(nib.Nifti1Image(generated_3D, orig_affine), voided_path)
        # Mask volume (channel 1)
        mask_raw = batch[0, 1, :, :, :].cpu().numpy().astype(np.float32)
        mask_path = os.path.join(inputs_dir, f"{base}_mask.nii.gz")
        nib.save(nib.Nifti1Image(mask_raw, orig_affine), mask_path)

        if not padded_range:
            nib.save(nib.Nifti1Image(generated_3D, orig_affine), out_file)
            continue

        # -----------------------------------------------------------------
        # build overlapping windows of fixed size to ensure full pseudo‑3D context
        # -----------------------------------------------------------------
        window_size = args.subbatch
        stride = max(1, window_size // 2)
        ext_min = padded_range[0]
        ext_max = padded_range[-1]
        # clamp to ensure windows fit within the volume
        max_start = max(depth - window_size, 0)
        windows = []  # list[list[int]] of slice indices per window
        seen = set()
        for logical_start in range(ext_min, ext_max + 1, stride):
            # determine valid window start
            start_idx = min(max(logical_start, 0), max_start)
            win = list(range(start_idx, start_idx + window_size))
            key = tuple(win)
            if key in seen:
                continue
            seen.add(key)
            windows.append(win)
        # -----------------------------------------------------------------
        # ensure every mask slice is included in at least one window
        # -----------------------------------------------------------------
        if orig_slices:
            covered = set(s for w in windows for s in w)
            missing = set(orig_slices) - covered
            if missing:
                logger.log(f"Warning: windows do not cover mask slices {sorted(missing)}, adding individual windows for them")
                for s in sorted(missing):
                    # center a window around slice s
                    half = window_size // 2
                    start_idx = min(max(s - half, 0), max_start)
                    win = list(range(start_idx, start_idx + window_size))
                    key = tuple(win)
                    if key not in seen:
                        seen.add(key)
                        windows.append(win)

        # -----------------------------------------------------------------
        # accumulate predictions for lesion slices only
        # -----------------------------------------------------------------
        # build prediction accumulator for each original lesion slice
        pred_accum = {s: [] for s in orig_slices}
        # load and clarify mask: ensure strict binary, optionally dilate boundary
        mask_vol = batch[0, 1].cpu().numpy().astype(bool)
        # if scipy available, dilate mask boundaries (in-plane) to avoid faint edges
        if binary_dilation is not None:
            mask_dil = np.zeros_like(mask_vol)
            for z in range(mask_vol.shape[2]):
                mask_dil[:, :, z] = binary_dilation(mask_vol[:, :, z], structure=np.ones((3, 3), bool))
            mask_vol = mask_dil
        device = dist_util.dev()
        # select the appropriate one-shot sampling function
        sample_fn = (
            diffusion.p_sample_loop_known
            if not args.use_ddim
            else diffusion.ddim_sample_loop_known
        )

        for win in windows:
            frames = [batch[..., si] for si in win]
            out_batch = th.stack(frames).squeeze(1).to(device)
            noise_chan = th.randn_like(out_batch[:, :1, ...])
            inp = th.cat((out_batch, noise_chan), dim=1)

            if (
                args.save_intermediate_interval > 0
                and args.save_intermediate_dir
                and not args.use_ddim
            ):
                # -----------------------------------------------------------------
                # progressive sampling for intermediate visualisation
                # -----------------------------------------------------------------
                prog = diffusion.p_sample_loop_progressive(
                    model,
                    (len(win), 3, args.image_size, args.image_size),
                    noise=inp,
                    clip_denoised=args.clip_denoised,
                    model_kwargs={},
                    device=device,
                    progress=True,
                )
                last_pred = None
                for step_idx, out_dict in enumerate(prog):
                    last_pred = out_dict["pred_xstart"]
                    if step_idx % args.save_intermediate_interval == 0:
                        # extract only the intensity channel (channel 0)
                        sa_int = last_pred.clone().detach()[:, 0].cpu().numpy()
                        tmp_vol = generated_3D.copy()
                        for idx, slice_idx in enumerate(win):
                            if slice_idx in orig_slices:
                                m2d = mask_vol[:, :, slice_idx]
                                tmp_vol[:, :, slice_idx][m2d] = sa_int[idx][m2d]
                        fn = os.path.join(
                            args.save_intermediate_dir,
                            f"{base}_w{win[0]}-{win[-1]}_step_{step_idx}.nii.gz",
                        )
                        nib.save(nib.Nifti1Image(tmp_vol, orig_affine), fn)
                        # -----------------------------------------------------------------
                        # log inpainting coverage at this intermediate step for current window
                        # -----------------------------------------------------------------
                        try:
                            mask_total = int(mask_vol.sum())
                            # predicted voxels in tmp_vol where mask is true
                            pred_vox = int((mask_vol & (tmp_vol != 0)).sum())
                            missing = mask_total - pred_vox
                            missing_slices = [s for s in orig_slices if np.any(mask_vol[:, :, s] & (tmp_vol[:, :, s] == 0))]
                            logger.log(f"[Intermediate] window {win[0]}-{win[-1]} step {step_idx}: "
                                       f"{pred_vox}/{mask_total} voxels predicted, {missing} missing; "
                                       f"missing slices: {missing_slices}")
                        except Exception:
                            logger.log(f"[Intermediate] window {win[0]}-{win[-1]} step {step_idx}: could not compute coverage stats")
                # after loop, use last_pred as final sample for window
                # extract intensity channel
                sam_np = last_pred[:, 0].cpu().numpy()
            else:
                # plain (non‑progressive) sampling
                sample_fn = (
                    diffusion.p_sample_loop_known
                    if not args.use_ddim
                    else diffusion.ddim_sample_loop_known
                )
                sample, _, _ = sample_fn(
                    model,
                    (len(win), 3, args.image_size, args.image_size),
                    inp,
                    clip_denoised=args.clip_denoised,
                    model_kwargs={},
                    progress=True,
                )
                # extract intensity channel
                sam_np = sample[:, 0].cpu().numpy()

            # collect predictions for averaging
            for idx, slice_idx in enumerate(win):
                if slice_idx in pred_accum:
                    pred_accum[slice_idx].append(sam_np[idx])

        # -----------------------------------------------------------------
        # fallback for any slices that got no predictions
        # -----------------------------------------------------------------
        missing = [s for s, preds in pred_accum.items() if not preds]
        if missing:
            logger.log(f"Missing predictions for slices {missing}, running fallback sampling...")
            for s in missing:
                # build local window around slice s
                ext_min = max(0, s - args.range_padding)
                ext_max = min(depth - 1, s + args.range_padding)
                win_s = list(range(ext_min, ext_max + 1))
                # prepare batch for this window
                frames_s = [batch[..., si] for si in win_s]
                out_batch_s = th.stack(frames_s).squeeze(1).to(device)
                noise_s = th.randn_like(out_batch_s[:, :1, ...])
                inp_s = th.cat((out_batch_s, noise_s), dim=1)
                # run one-shot known sampling
                sample_s, _, _ = sample_fn(
                    model,
                    (len(win_s), 3, args.image_size, args.image_size),
                    inp_s,
                    clip_denoised=args.clip_denoised,
                    model_kwargs={},
                    progress=False,
                )
                sam_s = sample_s[:, 0].cpu().numpy()
                # pick prediction corresponding to slice s
                idx_local = win_s.index(s)
                pred_accum[s].append(sam_s[idx_local])
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # merge overlapping predictions (average)
        # -----------------------------------------------------------------
        for slice_idx, preds in pred_accum.items():
            if not preds:
                continue
            avg_pred = np.mean(np.stack(preds, axis=0), axis=0)
            lesion_mask_2d = mask_vol[:, :, slice_idx]
            generated_3D[:, :, slice_idx][lesion_mask_2d] = avg_pred[lesion_mask_2d]

        # -----------------------------------------------------------------
        # log inpainting coverage before fallback
        # -----------------------------------------------------------------
        try:
            mask_total = int(mask_vol.sum())
            predicted = int(np.sum(mask_vol & (generated_3D != 0)))
            missing = mask_total - predicted
            missing_slices = [s for s in orig_slices if np.any(mask_vol[:, :, s] & (generated_3D[:, :, s] == 0))]
            logger.log(f"Inpainting coverage before fallback: {predicted}/{mask_total} voxels predicted, {missing} missing; missing slices: {missing_slices}")
        except Exception:
            logger.log("Could not compute inpainting coverage statistics")

        # -----------------------------------------------------------------
        # fill any remaining zeroed voxels in lesion region with original healthy intensities
        try:
            orig_img = batch[0, 2].cpu().numpy()
            # mask of voxels still zero and within dilated lesion
            zero_mask = (generated_3D == 0)
            fill_mask = mask_vol & zero_mask
            generated_3D[fill_mask] = orig_img[fill_mask]
        except Exception:
            pass
        # -----------------------------------------------------------------
        # write result
        # -----------------------------------------------------------------
        nib.save(nib.Nifti1Image(generated_3D, orig_affine), out_file)

    print("Sampling complete.")


# -----------------------------------------------------------------------------
# argument parser
# -----------------------------------------------------------------------------

def create_argparser():
    defaults = dict(
        data_dir="../FCD2/Test",
        log_dir="sample_log",
        adapted_samples="",  # unused for FCD2
        subbatch=16,
        clip_denoised=True,
        batch_size=1,
        use_ddim=False,
        model_path="log/xemasavedmodel_0.9999_050000.pt",
        range_padding=2,  # slices of context added on each side
        save_intermediate_dir="",
        save_intermediate_interval=0,
    )
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

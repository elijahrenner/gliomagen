import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from ddpm import Unet3D, Trainer, GaussianDiffusion_Nolatent
import hydra
from omegaconf import DictConfig, OmegaConf
from get_dataset.get_dataset import get_train_dataset
import torch
from ddpm.unet import UNet
import torch.nn as nn
import wandb
import torchio as tio

@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    wandb.init(
        project=cfg.wandb.project if 'wandb' in cfg and 'project' in cfg.wandb else "default_project",
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    torch.cuda.set_device(cfg.model.gpus)
    data_type = cfg.dataset.data_type.lower()
    if data_type not in ['lidc', 'emidec', 'fcd2']:
        raise ValueError("Wrong data type")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            cond_dim=cfg.model.cond_dim,
        )
    elif cfg.model.denoising_fn == 'UNet':
        model = UNet(
            in_ch=cfg.model.diffusion_num_channels,
            out_ch=cfg.model.diffusion_num_channels,
            spatial_dims=3
        )
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    model = nn.DataParallel(model)

    diffusion = GaussianDiffusion_Nolatent(
        model,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
        device=device,
        data_type=data_type
    ).to(device)

    train_dataset, *_ = get_train_dataset(cfg)
    
    # PRE-TRAINING TEST: Save one transformed training example for analysis
    pretrain_folder = os.path.join(cfg.model.results_folder, "PRE-TRAINING_TEST")
    os.makedirs(pretrain_folder, exist_ok=True)
    sample = train_dataset[0]
    # Save the transformed image volume
    img = tio.ScalarImage(tensor=sample['data'], affine=sample.get('affine'))
    gt_name = sample.get('GT_name', "sample.nii.gz")
    img_path = os.path.join(pretrain_folder, gt_name)
    img.save(img_path)
    print(f"Saved pre-training transformed image to: {img_path}")
    # Save the transformed mask if available
    if 'label' in sample:
         label = tio.LabelMap(tensor=sample['label'], affine=sample.get('affine'))
         label_name = gt_name.replace(".nii.gz", "_mask.nii.gz")
         label_path = os.path.join(pretrain_folder, label_name)
         label.save(label_path)
         print(f"Saved pre-training transformed mask to: {label_path}")

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        train_batch_size=cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        device=device,
    )

    if cfg.model.load_milestone and isinstance(cfg.model.load_milestone, str):
        trainer.load(cfg.model.load_milestone)
    else:
        # Auto-load the most recent checkpoint in the results folder, if any.
        ckpt_files = [f for f in os.listdir(cfg.model.results_folder) if f.endswith('.pt')]
        if ckpt_files:
            latest = max(ckpt_files, key=lambda f: int(f.split('-')[-1].split('.')[0]))
            latest_path = os.path.join(cfg.model.results_folder, latest)
            print(f"Auto-loading latest checkpoint: {latest_path}")
            trainer.load(latest_path)
        else:
            print("No checkpoint found. Starting training from scratch.")

    trainer.train()


if __name__ == '__main__':
    run()

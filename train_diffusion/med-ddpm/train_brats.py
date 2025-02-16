import os
import random
import logging
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from diffusion_model.unet_brats import create_model
from diffusion_model.trainer_brats import GaussianDiffusion, Trainer
from nibabel import as_closest_canonical  # For reorienting images
import matplotlib.pyplot as plt
import yaml

# ----------------------------------------
# Load configuration from YAML file
# ----------------------------------------
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ----------------------------------------
# Configure Logging
# ----------------------------------------
logging.basicConfig(
    filename="train.log",          # Log file name
    filemode="a",                  # Append mode
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# ----------------------------------------
# Set Environment Variables and Seeds
# ----------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ----------------------------------------
# Build Directory Paths Dynamically
# ----------------------------------------
base_dir = "../../data/"  # Base directory for your processed data
output_root = config["dataset"]["output_root"]
modalities = config["dataset"]["modalities"]

# Segmentation directories (fixed names)
train_seg_dir = os.path.join(base_dir, output_root, "train_seg")
val_seg_dir   = os.path.join(base_dir, output_root, "val_seg")

# Build modality directories for training and validation dynamically
train_modality_dirs = {}
val_modality_dirs   = {}
for mod in modalities:
    train_modality_dirs[mod] = os.path.join(base_dir, output_root, f"train_{mod}")
    val_modality_dirs[mod]   = os.path.join(base_dir, output_root, f"val_{mod}")

# ----------------------------------------
# Read Hyperparameters from Config
# ----------------------------------------
model_config    = config["model"]
training_config = config["training"]
preproc_config  = config["preprocessing"]

input_size         = tuple(model_config["input_size"])
num_channels       = model_config["num_channels"]
num_res_blocks     = model_config["num_res_blocks"]
num_class_labels   = model_config["num_class_labels"]
with_condition     = model_config["with_condition"]
resume_weight      = model_config["resume_weight"]
resume_step_offset = model_config["resume_step_offset"]
out_channels       = model_config["out_channels"]

train_lr = float(training_config["train_lr"])
batchsize            = training_config["batch_size"]
epochs               = training_config["epochs"]
timesteps            = training_config["timesteps"]
save_and_sample_every= training_config["save_and_sample_every"]
freeze_encoder_at_step = training_config["freeze_encoder_at_step"]

early_stopping_patience = training_config.get("early_stopping_patience", 20000)
early_stopping_delta    = training_config.get("early_stopping_delta", 0.0)
val_interval_steps      = training_config.get("val_interval_steps", 5000)
num_val_batches         = training_config.get("num_val_batches", 10)

# ----------------------------------------
# Data Transforms
# ----------------------------------------
class OneHotMask:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, mask):
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        one_hot = F.one_hot(mask, num_classes=self.num_classes + 1)
        one_hot = one_hot[..., 1:].permute(3, 0, 1, 2).float()  # [C, H, W, D]
        return one_hot

class ToTensorMask:
    def __call__(self, mask):
        return torch.from_numpy(mask).long()

class ToTensorImage:
    def __call__(self, image):
        return torch.from_numpy(image).float()

def sample_conditions(dataset, batch_size):
    indices = random.sample(range(len(dataset)), batch_size)
    conditions = [dataset[i]['input'] for i in indices]
    return torch.stack(conditions, dim=0)

mask_transform = transforms.Compose([
    ToTensorMask(),
    OneHotMask(num_class_labels),
])

image_transform = transforms.Compose([
    ToTensorImage(),
    transforms.Lambda(lambda t: t.unsqueeze(0)),  # Add channel dimension
])

LOW_QUALITY_MASKS = set([])

def prefix_from_seg_filename(seg_filename: str) -> str:
    base = seg_filename.replace(".nii.gz", "").replace(".nii", "")
    return base.replace("_seg", "")

# ----------------------------------------
# Flexible Dataset Class for Arbitrary Modalities
# ----------------------------------------
class BraTSMultimodalDataset(Dataset):
    def __init__(self, seg_dir, modality_dirs, transform=None, augment=False):
        """
        Parameters:
          seg_dir: Directory containing segmentation masks.
          modality_dirs: Dictionary mapping modality name to its directory.
        """
        self.seg_dir = seg_dir
        self.modality_dirs = modality_dirs  # e.g., {"t1n": "...", "t1c": "...", "t2w": "...", "t2f": "..."}
        self.modalities = list(modality_dirs.keys())
        
        # Collect segmentation filenames
        self.seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith((".nii", ".nii.gz"))])
        self.seg_files = [f for f in self.seg_files if f not in LOW_QUALITY_MASKS]
        
        # Ensure that each segmentation file has a corresponding file in each modality directory
        valid_segs = []
        for seg_file in self.seg_files:
            prefix = prefix_from_seg_filename(seg_file)
            valid = True
            for mod in self.modalities:
                candidate = prefix + f"_{mod}.nii.gz"
                if not os.path.exists(os.path.join(self.modality_dirs[mod], candidate)):
                    valid = False
                    break
            if valid:
                valid_segs.append(seg_file)
        self.seg_files = valid_segs
        
        logging.info(f"After filtering, total valid samples: {len(self.seg_files)}")
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.seg_files)

    def __getitem__(self, idx):
        seg_file = self.seg_files[idx]
        seg_path = os.path.join(self.seg_dir, seg_file)
        seg_img = nib.load(seg_path)
        seg_img = as_closest_canonical(seg_img)
        seg = np.array(seg_img.dataobj)
        affine = seg_img.affine

        modality_tensors = []
        prefix = prefix_from_seg_filename(seg_file)
        for mod in self.modalities:
            modality_filename = prefix + f"_{mod}.nii.gz"
            modality_path = os.path.join(self.modality_dirs[mod], modality_filename)
            mod_img = nib.load(modality_path)
            mod_img = as_closest_canonical(mod_img)
            mod_data = np.array(mod_img.dataobj)
            mod_tensor = image_transform(mod_data)
            modality_tensors.append(mod_tensor)
        
        if self.transform:
            seg = self.transform(seg)
        
        # Concatenate modality tensors along the channel dimension
        inputs = torch.cat(modality_tensors, dim=0)
        return {
            "input": seg,
            "target": inputs,
            "affine": affine
        }

# ----------------------------------------
# Initialize Datasets and DataLoaders
# ----------------------------------------
train_dataset = BraTSMultimodalDataset(
    seg_dir=train_seg_dir,
    modality_dirs=train_modality_dirs,
    transform=mask_transform,
    augment=False
)

val_dataset = BraTSMultimodalDataset(
    seg_dir=val_seg_dir,
    modality_dirs=val_modality_dirs,
    transform=mask_transform,
    augment=False
)

train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)

logging.info(f"Total training samples: {len(train_dataset)}")
logging.info(f"Total validation samples: {len(val_dataset)}")

in_channels = num_class_labels + out_channels if with_condition else out_channels

# ----------------------------------------
# Model Creation
# ----------------------------------------
try:
    model = create_model(
        image_size=input_size[0],
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        in_channels=in_channels,
        out_channels=out_channels
    ).cuda()
    logging.info("Model created successfully.")
except TypeError as e:
    logging.error(f"Error in create_model: {e}")
    raise e

# ----------------------------------------
# Freeze Encoder Logic for Fine-Tuning
# ----------------------------------------
def freeze_encoder(model, current_step, freeze_at_step):
    if current_step >= freeze_at_step:
        for param in model.input_blocks.parameters():
            param.requires_grad = False
        for param in model.middle_block.parameters():
            param.requires_grad = False

# Ensure the decoder remains trainable
for param in model.output_blocks.parameters():
    param.requires_grad = True

# ----------------------------------------
# Diffusion Model Setup
# ----------------------------------------
diffusion = GaussianDiffusion(
    denoise_fn=model,
    image_size=input_size[0],
    depth_size=input_size[2],
    timesteps=timesteps,
    loss_type="l1",
    with_condition=with_condition,
    channels=out_channels
).cuda()

if resume_weight:
    try:
        weight = torch.load(resume_weight, map_location="cuda")
        diffusion.load_state_dict(weight["ema"], strict=False)
        logging.info(f"Model loaded from: {resume_weight}")
    except Exception as e:
        logging.error(f"Failed to load weights: {e}")
        raise e

# ----------------------------------------
# Trainer Initialization
# ----------------------------------------
trainer = Trainer(
    diffusion_model=diffusion,
    train_loader=train_loader,
    val_dataset=val_dataset,
    image_size=input_size[0],
    depth_size=input_size[2],
    train_batch_size=batchsize,
    train_lr=train_lr,
    train_num_steps=epochs,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    fp16=False,
    with_condition=with_condition,
    save_and_sample_every=save_and_sample_every,
    results_folder="results",
    resume_step_offset=resume_step_offset,
    val_interval_steps=val_interval_steps,
    num_val_batches=num_val_batches,
    early_stopping_patience=early_stopping_patience,
    early_stopping_delta=early_stopping_delta,
    freeze_encoder_at_step=freeze_encoder_at_step
)

# ----------------------------------------
# Start Training
# ----------------------------------------
trainer.train()
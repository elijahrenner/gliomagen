# GliomaGen

Elijah Renner
 
![image](figures/diffusion%20processes%20with%20mask.png)

[Paper Link](#) | [Weights](#) | [Dataset](#)

## Abstract

The scarcity of labeled post-treatment glioma MR images limits effective automatic segmentation of key features in brain MR images. Addressing this issue, GliomaGen is introduced, an anatomically informed generative diffusion model that uses a modified Med-DDPM structure to create high-quality MR images from anatomical masks. GliomaGen takes four modalities and six segmentation labels, including a new head area, as input. The developed GliomaGen pipeline augments existing masks to expand the BraTS 2024 Post-Treatment Glioma dataset by 2124 masks, which are later used to synthesize the largest BraTS 2024 Adult Post-Treatment Glioma derivative synthetic dataset $(N=2124)$. Evaluations of GliomaGen with quantitative metrics MS-SSIM, FID, and KID show high fidelity, particularly for t1c (FID: 55.2028 ± 3.7446) and t2w (FID: 54.9974 ± 3.2271) modalities. Segmentation tests with nnU-Net show hybrid training matches real-data performance, but inconsistencies and noise in generated volumes prevented state-of-the-art segmentation from being achieved. These findings show the potential of conditional diffusion models to address data constraints in the BraTS 2024 Adult Post-Treatment Glioma context, and also prompt further iteration on the GliomaGen pipeline.

## Setup

### YAML Configuration

To seamlessly train and synthesize data with GliomaGen, please update the `config.yaml` file, which contains all hyperparameters and settings inherited by the code.  

### Packages

Install all packages with 

```
pip install -r requirements.txt
```

### Hardware

Depending on your configuration (volume dimensions, channels, etc.), memory requirements will vary. However, generally, it is recommended that you train with > 48GB of VRAM. All experiments described in the paper with volume dimensions $(192, 192, 144)$ were carried out on a NVIDIA A100.

## All Things Data

Based on Med-DDPM, GliomaGen has applications to several imaging domains, particularly those in the BraTS challenge. To use BraTS 2024 Adult Post-Treatment Glioma as described in the paper, follow these instructions, which are easily adaptable to other domains:

### BraTS 2024 Adult Post-Treatment Glioma Usage

To collect the data used for training GliomaGen,

1. Create an account and gain access to the [BraTS 2024 Challenge](https://www.synapse.org/Synapse:syn53708249).
2. Install the Synapse CLI: `pip install synapseclient`
3. Gain a personal access token in account settings and login to the CLI via `synapse login -p $MY_SYNAPSE_TOKEN`.
4. Navigate to the files section of the BraTS 2024 Challenge and find the ID of the desired training dataset (syn60086071 for BraTS 2024 Adult Post-Treatment Glioma).
5. Download the data using `synapse get $ID` (`synapse get syn60086071`). Use the personal access token from before as the auth token.
6. Move the download archive into the `data/BraTS2024-GLI` folder and extract it. Rename the folder containing all of the cases to `training`.

Next, to prepare the data for GliomaGen, follow the instructions in `data/prepare_gliomagen_data.ipynb`. 

## Training

Ensure `config.yaml` contains the desired hyperparameters (e.g., iterations, LR schedule, timesteps, etc.) and dataset directories. Then, train the model by running

```
cd train_diffusion/med-ddpm
python train_brats.py
```

## Dataset Synthesis

## nnU-Net Evaluation

## BibTeX
# GliomaGen
 
![image](figures/diffusion%20processes%20with%20mask.png)

[📝 Paper](figures/GliomaGen.pdf) | [🤗 Weights](https://huggingface.co/elijahrenner/gliomagen) | [🤗 BraTS 2024 Adult Post Treatment Glioma-Synthetic](https://huggingface.co/datasets/elijahrenner/brats2024-aptg-synthetic)

## Abstract

The scarcity of labeled post-treatment glioma MR images limits effective automatic segmentation of key features in brain MR images. Addressing this issue, GliomaGen is introduced, an anatomically informed generative diffusion model that uses a modified Med-DDPM structure to create high-quality MR images from anatomical masks. GliomaGen takes four modalities and six segmentation labels, including a new head area, as input. The developed GliomaGen pipeline augments existing masks to expand the BraTS 2024 Post-Treatment Glioma dataset by 2124 masks, which are later used to synthesize the largest BraTS 2024 Adult Post-Treatment Glioma derivative synthetic dataset $(N=2124)$. Evaluations of GliomaGen with quantitative metrics MS-SSIM, FID, and KID show high fidelity, particularly for t1c (FID: 55.2028 ± 3.7446) and t2w (FID: 54.9974 ± 3.2271) modalities. Segmentation tests with nnU-Net show hybrid training matches real-data performance, but inconsistencies and noise in generated volumes prevented state-of-the-art segmentation from being achieved. These findings show the potential of conditional diffusion models to address data constraints in the BraTS 2024 Adult Post-Treatment Glioma context, and also prompt further iteration on the GliomaGen pipeline.

## Setup

### YAML Configuration

To seamlessly train and synthesize data with GliomaGen, please update the `config.yaml` file, which contains all hyperparameters and settings inherited by the code.  

### Packages

Install all packages with ```
pip install -r requirements.txt```

### Hardware

Depending on your configuration (volume dimensions, channels, etc.), memory requirements will vary. However, generally, it is recommended that you train with > 48GB of VRAM. All experiments described in the paper with volume dimensions $(192, 192, 144)$ were carried out on a NVIDIA A100 80GB.

## All Things Data

Based on [Med-DDPM](https://github.com/mobaidoctor/med-ddpm), GliomaGen has applications to several imaging domains, particularly those in the BraTS challenge. To use BraTS 2024 Adult Post-Treatment Glioma as described in the paper, follow these instructions, which are easily adaptable to other domains:

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

As mentioned before, it's crucial `config.yaml` contains the desired hyperparameters (e.g., iterations, LR schedule, timesteps, etc.) and dataset directories tailored to your dataset. Then, train the model by running
```
cd train_diffusion/med-ddpm
python train.py
```

When finished, you may run inference on all samples in the `val_seg` folder of your dataset by running

```
python sample.py
```

which will output all generated volumes to `data/{your dataset}/{your transformed dataset}/val_pred`.

To evaluate the generated images against ground truth, run 

```
cd .. # to train_diffusion
python eval.py
```

which will print MS-SSIM, FID, and KID metrics.

## Dataset Synthesis

Start by augmenting the original masks using `synthesize_dataset/generate_anatomical_maps.ipynb` to create `data/{your dataset}-Synthetic/seg`. Then, ensure there is a model named `synthesize_dataset/final_model.pt`. You may then run inference on all augmented masks with 

```
cd synthesize_dataset
python generate_synthetic_dataset.py
```

which will populate `data/{yourdataset}-Synthetic` with the each of the corresponding modalities in `data/{yourdataset}-Synthetic/seg`.

## nnU-Net Evaluation

To evaluate the quality of a generated synthetic dataset, you may train [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), a SoTA segmentation model, on different configurations. The configurations discussed in the paper are provided in `validate_dataset/prepare_data_nnunet.ipynb`. To train nnU-Net, follow these steps:

Install nnU-Net with

```
cd validate_dataset
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

before setting these environment variables:

```
export nnUNet_raw="nnUNet_raw"
export nnUNet_preprocessed="nnUNet_preprocessed"
export nnUNet_results="nnUNet_results"
```

Now, ensure data is properly prepared in `validate_dataset/nnUNet/nnUNet_raw` with `prepare_data_nnunet.ipynb`.

Then, preprocess and verify the data using the nnU-Net CLI:

```
nnUNetv2_plan_and_preprocess -d {your dataset ID} --verify_dataset_integrity
```

and, finally, train nnU-Net using

```
nnUNetv2_train {your dataset ID} 3d_fullres all --npz
```

**NOTE 1**: The above instructions simply reproduce the tests performed in the paper. Exploring the nnU-Net repository may be helpful for engineering a more effective procedure, and the one used here is quite bare bones.

**NOTE 2**: if you encounter issues related OPENBLAS or a C compiler, try

```
export OPENBLAS_NUM_THREADS=8

sudo apt-get update
sudo apt-get install build-essential
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
```

To evaluate the performance of different nnU-Net configurations, you may run inference on the validation dataset:

```
nnUNetv2_predict \
-i nnUNet_raw/{path_to_test_or_val_images} \
-o nnUNet_predictions \
-d {dataset_folder_name (e.g., Dataset001_GliHeadMask)} \
-c 3d_fullres \
-f all
```

Then, you can use `validate_dataset/evaluate_nnunet.ipynb` to produce several quantiative metrics (accuracy, dice, F1, etc.) as described in the paper.

## To Do

- [X] Release code
- [ ] Release weights
- [ ] Release dataset
- [ ] Put paper on arXiv
- [ ] Train on other datasets

## BibTeX

```
@ARTICLE{gliomagen,
	author = {Elijah Renner},
	title = {Unlimited Post-Treatment Glioma MR Images via Conditional Diffusion},
	howpublished = {\url{https://github.com/elijahrenner/gliomagen}},
	year = {2025},
}
```




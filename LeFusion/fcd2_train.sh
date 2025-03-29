dataset=fcd2
diffusion_img_size=128
diffusion_depth_size=80
diffusion_num_channels=1
batch_size=2
test_txt_dir=data/FCD2/Pathological/test.txt
dataset_root_dir=data/FCD2
train_num_steps=50001
cond_dim=16
results_folder=LeFusion/LeFusion_Model/FCD2

python LeFusion/train/train.py \
    dataset=$dataset \
    model.diffusion_img_size=$diffusion_img_size \
    model.diffusion_depth_size=$diffusion_depth_size \
    model.diffusion_num_channels=$diffusion_num_channels \
    dataset.test_txt_dir=$test_txt_dir \
    dataset.root_dir=$dataset_root_dir \
    model.train_num_steps=$train_num_steps \
    model.batch_size=$batch_size \
    model.cond_dim=$cond_dim \
    model.results_folder=$results_folder \


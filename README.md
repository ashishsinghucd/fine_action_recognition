Fine-Grained Action Recognition with MMAction2This repository is a fork of the official MMAction2 framework, tailored specifically for fine-grained action recognition tasks. The goal is to provide a streamlined environment for fine-tuning state-of-the-art models like SlowFast, I3D, and PoseC3D on datasets such as FineGym and Diving48.This README serves as a practical guide to setting up the environment, preparing datasets, and running training and inference workflows.🚀 Quickstart: Fine-Tuning WorkflowThe core philosophy of MMAction2 is to modify configuration files rather than writing boilerplate code. Here's the standard workflow:Prepare Data: Organize your videos and create simple text-based annotation files.Choose a Config: Select a pre-existing config file for a model like SlowFast.Modify the Config: Adapt the config for your dataset (e.g., update class numbers and data paths).Train: Run the training script with your new config.Test & Infer: Evaluate your trained model's performance.🛠️ InstallationIt is highly recommended to use a Conda environment to manage the complex dependencies.1. Create Conda Environmentconda create --name mmaction-env python=3.9 -y
conda activate mmaction-env
2. Install PyTorchInstall a version of PyTorch compatible with your CUDA toolkit. Check the official PyTorch website for the correct command.# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
3. Install MMLab Dependencies (The Correct Way)To avoid common dependency conflicts, install the core MMLab packages in a specific order.# Install base engines
pip install -U openmim
mim install mmengine "mmcv>=2.0.0"

# Install the correct version of mmdet required by mmpose
pip install "mmdet<3.3.0"

# Install mmpose
pip install "mmpose>=1.0.0"
4. Install MMAction2Finally, clone this repository and install MMAction2.git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
pip install -v -e .
📂 Dataset PreparationYour model's performance depends heavily on how you structure your data.For RGB / Optical Flow ModelsVideo/Frame Folder: Place all your videos or extracted frames in a directory.Annotation Files: Create train.txt and val.txt files where each line maps a video file to its integer label. The path should be relative to the data_root you'll specify in the config.Example train.txt:abseiling/video_00003.mp4 0
abseiling/video_00004.mp4 0
diving/video_00010.mp4 1
...
For Skeleton-Based Models (PoseC3D)For skeleton-based models, you only need the annotation pickle file provided by MMAction2 (e.g., diving48_hrnet.pkl). The PoseDataset class will handle parsing it directly.🏋️‍♀️ Training a ModelHere’s how to fine-tune a SlowFast model on the Diving48 dataset.1. Choose and Copy a Config Filecp configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py \
   configs/recognition/slowfast/slowfast_r50_diving48.py
2. Modify slowfast_r50_diving48.pyUpdate num_classes: Change the number of output classes in the model's head.model = dict(
    cls_head=dict(
        type='SlowFastHead',
        num_classes=48,  # Changed from 400
        ...))
Update Dataset Paths: Point to your data and annotation files.dataset_type = 'VideoDataset'
data_root = 'data/diving48/videos/'
data_root_val = 'data/diving48/videos/'
ann_file_train = 'data/diving48/train.txt'
ann_file_val = 'data/diving48/val.txt'
Load Pre-trained Weights: Add this line to leverage transfer learning. This is critical for good performance.load_from = '[https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth)'
3. Run TrainingLaunch the training process from the main mmaction2 directory.python tools/train.py \
    configs/recognition/slowfast/slowfast_r50_diving48.py \
    --work-dir work_dirs/slowfast_diving48
Checkpoints and logs will be saved to the work_dirs/slowfast_diving48 folder.🧪 Testing and InferenceTo evaluate your model's accuracy on the validation or test set, use the tools/test.py script.python tools/test.py \
    configs/recognition/slowfast/slowfast_r50_diving48.py \
    work_dirs/slowfast_diving48/best_acc_top1_epoch_10.pth \
    --eval top_k_accuracy
✨ Included UtilitiesThis repository includes custom scripts to help with data processing and visualization.Pose Visualizer (pose_visualizer.py)This script overlays 2D pose skeletons from an MMAction2 annotation pickle file onto the corresponding video. This is extremely useful for verifying that your pose data is correct.Usage:Fill out the config.ini file with the paths to your video, the .pkl annotation file, and the desired output path.Run the script: python pose_visualizer.py🙏 AcknowledgementsThis work is built entirely on the incredible open-source efforts of the MMAction2 team and the broader OpenMMLab community. Please cite their work if you use this repository in your research.

`
@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}
`
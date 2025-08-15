# This config fine-tunes an I3D R50 model on the HAA500 dataset.

# --- 1. Inherit from a base model and default settings ---
# We inherit the I3D R50 model structure and default runtime settings.
_base_ = [
    '../../_base_/models/i3d_r50.py',
    '../../_base_/default_runtime.py'
]

# --- 2. Override the model's classification head ---
# This is the most IMPORTANT step for fine-tuning.
# We replace the final layer to match the number of classes in HAA500.
model = dict(
    cls_head=dict(
        # The number of classes in your new dataset.
        num_classes=500,
        # The number of input features from the I3D backbone.
        in_channels=2048,
    ))

# --- 3. Dataset settings ---
# Define the type of dataset and paths to your data and annotation files.
dataset_type = 'VideoDataset'
# The root directory where your video files are stored.
data_root = '/home/ashisig/Research/Data/HAA500/videos'
data_root_val = '/home/ashisig/Research/Data/HAA500/videos'
# The split number to use.
split = 1
# Annotation files mapping video paths to their labels.
ann_file_train = f'/home/ashisig/Research/Data/HAA500/splits/train_split_{split}_videos.txt'
ann_file_val = f'/home/ashisig/Research/Data/HAA500/splits/val_split_{split}_videos.txt'

# --- 4. Data processing pipelines ---
# These are standard CNN pipelines for video.
file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # I3D typically uses clips of 32 frames.
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        # Use 10 clips and 3 crops for robust testing.
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# --- 5. Dataloader configuration ---
train_dataloader = dict(
    batch_size=8,  # Adjust based on your GPU memory. 8 is a good start for I3D.
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val, # Test on the validation set
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

# --- 6. Evaluation settings ---
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# --- 7. Training loop, optimizer, and learning rate schedule ---
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer: Standard SGD is used for fine-tuning CNNs.
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

# Learning rate schedule: A step-based decay is common for I3D.
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=100, # Should match max_epochs
        by_epoch=True,
        milestones=[40, 80], # Decay LR at these epochs
        gamma=0.1)
]

# --- 8. Pre-trained Model ---
# This checkpoint contains an I3D model that has been trained
# in two stages:
#   1. Its ResNet backbone was pre-trained on ImageNet.
#   2. The full I3D model was then pre-trained on Kinetics-400.
# This provides a very powerful initialization for your HAA500 task.
load_from = 'https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth'

# --- 9. Runtime settings ---
# Enable automatic learning rate scaling based on the number of GPUs and batch size.
auto_scale_lr = dict(enable=True, base_batch_size=64)

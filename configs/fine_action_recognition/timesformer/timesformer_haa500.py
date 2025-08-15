# This config fine-tunes a TimeSformer model on the HAA500 dataset.

# --- 1. Inherit from a base model and default settings ---
# We inherit the TimeSformer model structure and default runtime settings.
# The base model defines the transformer architecture.
_base_ = [
    '../../_base_/models/timesformer_base.py',
    '../../_base_/default_runtime.py'
]

# --- 2. Override the model's classification head ---
# This is the most IMPORTANT step for fine-tuning.
# We replace the final layer to match the number of classes in HAA500.
model = dict(
    cls_head=dict(
        # The number of classes in your new dataset.
        num_classes=500,
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
# These pipelines are standard for Vision Transformer models.
file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # TimeSformer typically uses fewer, more spaced-out frames.
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        # Use 3 clips for more robust testing.
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# --- 5. Dataloader configuration ---
train_dataloader = dict(
    batch_size=8,  # Adjust based on your GPU memory. 8 is a good start for ViT models.
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
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer: AdamW is standard for Transformer-based models.
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

# Learning rate schedule: A simple step-based decay.
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50, # Should match max_epochs
        by_epoch=True,
        milestones=[25, 45], # Decay LR at these epochs
        gamma=0.1)
]

# --- 8. Pre-trained Model ---
# This checkpoint contains a TimeSformer model that has been trained
# in two stages:
#   1. Its Vision Transformer backbone was pre-trained on ImageNet-21K.
#   2. The full TimeSformer model was then fine-tuned on Kinetics-400.
# This provides a very powerful initialization for your HAA500 task.
load_from = 'https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth'

# --- 9. Runtime settings ---
# Enable automatic learning rate scaling based on the number of GPUs and batch size.
auto_scale_lr = dict(enable=True, base_batch_size=64)

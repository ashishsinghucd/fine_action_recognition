# This config fine-tunes a SlowFast R50 model on the HAA500 dataset.

# --- 1. Inherit from a base model and default settings ---
# We inherit the SlowFast R50 model structure and default runtime settings.
_base_ = [
    '../../_base_/models/slowfast_r50.py',
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
# These pipelines are specific to the SlowFast model, preparing separate
# inputs for the Slow and Fast pathways.
file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    # Sample 32 frames for the Fast path and 8 frames for the Slow path.
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    # This step creates the two pathways for SlowFast.
    dict(type='SlowFastPackInputs'),
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
    dict(type='SlowFastPackInputs'),
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        # Use 10 clips for more robust testing.
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='SlowFastPackInputs'),
]

# --- 5. Dataloader configuration ---
train_dataloader = dict(
    batch_size=8,  # Adjust based on your GPU memory. 8 is a safe start.
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_sze=8,
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
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

# Learning rate schedule: A cosine annealing schedule is effective.
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=100, # Should match max_epochs
        eta_min=0,
        by_epoch=True,
    )
]

# --- 8. Pre-trained Model ---
# This is the crucial part for producing a strong baseline.
# This checkpoint contains a SlowFast model that has already been trained
# in two stages:
#   1. Its ResNet backbone was pre-trained on ImageNet (learning basic visuals).
#   2. The full SlowFast model was then pre-trained on Kinetics-400 (learning general actions).
# By loading this, you are transferring all of that knowledge to your HAA500 task.
load_from = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth'

# --- 9. Runtime settings ---
# Enable automatic learning rate scaling based on the number of GPUs and batch size.
auto_scale_lr = dict(enable=True, base_batch_size=64)

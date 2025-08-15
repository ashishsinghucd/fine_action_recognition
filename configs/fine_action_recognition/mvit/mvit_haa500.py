# This config fine-tunes an MViT V2 model on the HAA500 dataset.

# --- 1. Inherit from a base model and default settings ---
# We inherit the MViT V2 Base model structure and default runtime settings.
_base_ = [
    '../../_base_/models/mvit_v2_base.py',
    '../../_base_/default_runtime.py'
]

# --- 2. Override the model's classification head ---
# Replace the final layer to match the number of classes in HAA500.
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
# These pipelines are specific to Vision Transformer models like MViT.
# They include augmentations like RandAugment and random erasing.
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_std=0.5,
        aug_space='rand-m9-n2-mstd0.5'),
    dict(type='RandomErasing', erase_prob=0.25, mode='pixel'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=4, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# --- 5. Dataloader configuration ---
train_dataloader = dict(
    batch_size=4,  # MViT is very memory-intensive. Start with a small batch size.
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
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

# MViT uses the AdamW optimizer and a cosine annealing schedule.
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=1.6e-3, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_positional_encoding': dict(decay_mult=0.0),
            '.backbone.pos_embed_spatial': dict(decay_mult=0.0),
            '.backbone.pos_embed_temporal': dict(decay_mult=0.0),
            '.backbone.pos_embed_class': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=1, norm_type=2))

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        eta_min=1.6e-4,
        by_epoch=True,
        begin=5,
        end=100,
        convert_to_iter_based=True)
]

# --- 8. Pre-trained Model ---
# Load weights from an MViT V2 model pre-trained on Kinetics-400.
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-base-p16x4_32x2x1_kinetics400-rgb/mvit-base-p16x4_32x2x1_kinetics400-rgb_20220805-20275623.pth'

# --- 9. Runtime settings ---
# Enable automatic learning rate scaling based on the number of GPUs.
# The base_batch_size should match what was used for the original pre-training.
auto_scale_lr = dict(enable=True, base_batch_size=64)

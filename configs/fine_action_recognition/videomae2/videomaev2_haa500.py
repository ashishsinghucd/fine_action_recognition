# This config fine-tunes a VideoMAE V2 model on the HAA500 dataset.

# --- 1. Inherit from a base model and default settings ---
# We inherit the VideoMAE V2 Base model structure and default runtime settings.
# This particular base config is set up for Kinetics-400 pre-training.
_base_ = [
    '../../_base_/models/videomae_v2_base.py',
    '../../_base_/default_runtime.py'
]

# --- 2. Override the model's classification head ---
# This is the most IMPORTANT step for fine-tuning.
# We replace the final layer to match the number of classes in HAA500.
model = dict(
    cls_head=dict(
        # The number of classes in your new dataset.
        num_classes=500,
    ),
    # The training and testing settings for the model.
    train_cfg=dict(
        type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=5),
    test_cfg=dict(type='TestLoop'))

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
# These pipelines are specific to Vision Transformer models and include
# specialized augmentations like RandAugment and Mixup/Cutmix.
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),
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
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=4, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# --- 5. Dataloader configuration ---
train_dataloader = dict(
    batch_size=8,  # ViT models are memory-intensive. Start with a smaller batch size.
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

# --- 7. Optimizer and Learning Rate Schedule ---
# VideoMAE uses the AdamW optimizer and a cosine annealing schedule.
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1.5e-3, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }))

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
        eta_min=1e-6,
        by_epoch=True,
        begin=5,
        end=100,
        convert_to_iter_based=True)
]

# --- 8. Pre-trained Model ---
# Load weights from a model pre-trained on Kinetics-400.
# The filename provides details about the model and its training.
#
# - vit-base-p16: The architecture is a "base" size Vision Transformer
#   with a patch size of 16x16 pixels.
#
# - videomae-v2-kinetics-400-pre: The model is VideoMAE V2, and it was
#   pre-trained on the Kinetics-400 dataset.
#
# - 16x4x1: The sampling strategy during pre-training was 1 clip of 16 frames,
#   with a frame interval of 4.
#
# - kinetics-400-e100: Confirms the pre-training dataset and that it was
#   trained for 100 epochs.
#
# - 20230510: The date the checkpoint was released.
# - 25c73783: A unique hash for the checkpoint file.
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae-v2/vit-base-p16_videomae-v2-kinetics-400-pre_16x4x1_kinetics-400-e100_20230510-25c73783.pth'

# --- 9. Runtime settings ---
auto_scale_lr = dict(enable=True, base_batch_size=64)

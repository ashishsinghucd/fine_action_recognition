# This config fine-tunes a PoseC3D model on the HAA500 dataset.

# --- 1. Inherit from a base runtime config ---
# This provides default settings for logging, checkpointing, etc.
_base_ = '../../_base_/default_runtime.py'

# --- 2. Dataset Settings ---
# Define the type of dataset and the path to your annotation file.
dataset_type = 'PoseDataset'
# IMPORTANT: Replace this with the actual path to your HAA500 skeleton annotation pickle file.
ann_file = '/path/to/your/haa500_skeletons.pkl'

# --- 3. Data Processing Pipeline ---
# This pipeline is specific to skeleton data. It generates a "pseudo heatmap"
# from the raw keypoint coordinates, which is then fed into the 3D CNN.
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=True, left_kp=(5, 7, 9, 11, 13, 15), right_kp=(6, 8, 10, 12, 14, 16)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# --- 4. Dataloader Configuration ---
# Configure how data is loaded for training, validation, and testing.
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='train',  # Use the 'train' split defined inside the pickle file.
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='val',  # Use the 'val' split defined inside the pickle file.
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='val', # Test on the validation set.
        pipeline=test_pipeline,
        test_mode=True))

# --- 5. Model Configuration ---
# Define the PoseC3D model architecture.
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        in_channels=17,  # 17 keypoints from the COCO format.
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2)),
    cls_head=dict(
        type='I3DHead',
        # IMPORTANT: Change this to the number of classes in HAA500.
        num_classes=500,
        in_channels=512,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'))

# --- 6. Evaluation, Training, and Optimizer Settings ---
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=120, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0003))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=120, # Should match max_epochs
        by_epoch=True,
        convert_to_iter_based=True)
]

# --- 7. Pre-trained Model ---
# Load weights from a model pre-trained on a large skeleton dataset (NTU-60)
# This is crucial for getting good performance.
load_from = 'https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-3d7f3c41.pth'

# --- 8. Runtime Settings ---
# Enable automatic learning rate scaling for multi-GPU training.
auto_scale_lr = dict(enable=True, base_batch_size=16)

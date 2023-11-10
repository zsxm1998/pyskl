model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C3D',
        in_channels=17,
        base_channels=32,
        num_stages=3,
        temporal_downsample=False),
    cls_head=dict(
        type='EnergyEstimateHead',
        in_channels=256,
        loss_func=dict(type='L1Loss'),
        dropout=0.0),
    test_cfg=dict(average_clips='score')) #使用score而不是prob来避免将模型输入经过softmax
#pretrained = '/medical-data/zsxm/pretrained_weight/c3d_joint_ntu60xsub2d.pth'

dataset_type = 'PoseDataset'
ann_file = '/medical-data/zsxm/运动热量估计/eev_resized/clips/prefix_eem_x_delay.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='AllFrames', num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='AllFrames', num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='AllFrames', num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=True, left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='valid', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='test', pipeline=test_pipeline))
# optimizer
optimizer = dict(type='Lion', lr=0.0001, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=True, min_lr=0.000001)
total_epochs = 50
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['percentage_loss', 'l1_loss', 'mse_loss', 'corr'])
# runtime settings
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './zcvpr/um/posec3d/c3d'
auto_resume = False
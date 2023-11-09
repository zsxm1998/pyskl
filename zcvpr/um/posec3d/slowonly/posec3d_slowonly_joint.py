model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        in_channels=17,
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
        type='EnergyEstimateHead',
        in_channels=512,
        loss_func=dict(type='L1Loss'),
        dropout=0.0),
    test_cfg=dict(average_clips='score'))
dataset_type = 'PoseDataset'
ann_file = '/medical-data/zsxm/运动热量估计/eev_resized/clips/eem_x_delay.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='AllFrames', num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(
        type='Flip',
        flip_ratio=0.5,
        left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='AllFrames', num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='AllFrames', num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        with_kp=True,
        with_limb=False,
        double=True,
        left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=16,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='PoseDataset',
            ann_file=
            '/medical-data/zsxm/运动热量估计/eev_resized/clips/eem_x_delay.pkl',
            split='train',
            pipeline=[
                dict(type='AllFrames', num_clips=1),
                dict(type='PoseDecode'),
                dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
                dict(type='Resize', scale=(-1, 64)),
                dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
                dict(type='Resize', scale=(56, 56), keep_ratio=False),
                dict(
                    type='Flip',
                    flip_ratio=0.5,
                    left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
                    right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
                dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
                dict(type='FormatShape', input_format='NCTHW_Heatmap'),
                dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['imgs', 'label'])
            ])),
    val=dict(
        type='PoseDataset',
        ann_file='/medical-data/zsxm/运动热量估计/eev_resized/clips/eem_x_delay.pkl',
        split='valid',
        pipeline=[
            dict(type='AllFrames', num_clips=1),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(64, 64), keep_ratio=False),
            dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
            dict(type='FormatShape', input_format='NCTHW_Heatmap'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='PoseDataset',
        ann_file='/medical-data/zsxm/运动热量估计/eev_resized/clips/eem_x_delay.pkl',
        split='test',
        pipeline=[
            dict(type='AllFrames', num_clips=1),
            dict(type='PoseDecode'),
            dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
            dict(type='Resize', scale=(64, 64), keep_ratio=False),
            dict(
                type='GeneratePoseTarget',
                with_kp=True,
                with_limb=False,
                double=True,
                left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
                right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
            dict(type='FormatShape', input_format='NCTHW_Heatmap'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
optimizer = dict(type='Lion', lr=0.0001, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='CosineAnnealing', by_epoch=True, min_lr=1e-06)
total_epochs = 50
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1, metrics=['percentage_loss', 'l1_loss', 'mse_loss', 'corr'])
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './zcvpr/um/posec3d/slowonly'
auto_resume = False
dist_params = dict(backend='nccl')
gpu_ids = range(0, 1)

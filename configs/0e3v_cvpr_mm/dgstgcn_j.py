model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='DGSTGCN',
        gcn_ratio=0.125,
        gcn_ctr='T',
        gcn_ada='T',
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(layout='coco', mode='random', num_filter=8, init_off=.04, init_std=.02)),
    cls_head=dict(
        type='MMEnergyEstimateHead',
        in_channels=256,
        mode='GCN',
        loss_func=dict(type='L1Loss'),
        dropout=0.0),
    test_cfg=dict(average_clips='score'))

dataset_type = 'PoseDataset'
ann_file = '/medical-data/zsxm/运动热量估计/eev_resized/clips/eem_x_delay.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='AllFrames', num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'heart_rate', 'weight', 'height', 'age', 'sex', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint', 'heart_rate', 'weight', 'height', 'age', 'sex', 'label'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='AllFrames', num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'heart_rate', 'weight', 'height', 'age', 'sex', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint', 'heart_rate', 'weight', 'height', 'age', 'sex', 'label'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='AllFrames', num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'heart_rate', 'weight', 'height', 'age', 'sex', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint', 'heart_rate', 'weight', 'height', 'age', 'sex', 'label'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=16,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='valid'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test'))

# optimizer
optimizer = dict(type='Lion', lr=0.0001, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=True, min_lr=0.000001)
total_epochs = 50
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['percentage_loss', 'l1_loss', 'mse_loss', 'corr'])
# runtime settings
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './zcvpr/mm/dgstgcn/j'
auto_resume = False

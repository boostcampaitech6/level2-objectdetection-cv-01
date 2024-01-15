_base_ = [
    '../configs/_base_/datasets/coco_detection.py',
    '../configs/_base_/default_runtime.py'
]

"""
참고사항

각 섹션 별 주석으로 작성되어 있는 내용을 잘 확인하고 학습 바랍니다.
현재 작성된 내용이 swin-l_ddq의 default 내용이며, 변경된 내용은 DataLoader 부분의 Randresize 부분만 고정된
image size (1024, 1024)로 변경되었습니다.
현재 공유되는 현재 버전을 기반으로 각자 맡으신 아이디어를 실험하시면 되겠습니다.
"""

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa: E501

### Model ###
model = dict(
    type='DDQDETR',
    num_queries=900,  # num_matching_queries
    # ratio of num_dense queries to num_queries
    dense_topk_ratio=1.5,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    # encoder class name: DeformableDetrTransformerEncoder
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    # decoder class name: DDQTransformerDecoder
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DDQDETRHead',
        num_classes=10,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    dqs_cfg=dict(type='nms', iou_threshold=0.8),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))


#### Hook ####
# Test시 inference 결과를 이미지로 확인하고 싶으면 아래 주석 해제
# default_hooks = dict(visualization=dict(type="DetVisualizationHook",draw=True))

# custom hooks
custom_hooks = [dict(type='SubmissionHook')]


### Dataset ###
data_root = '/data/ephemeral/home/dataset/'
metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}
image_size = (1024, 1024)

# RandAugment Color Space
color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

# RandomResize, RandomCrop, RandAugment 를 필요시 주석 해제
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # dict(
    #     type='RandomResize',
    #     scale=image_size,
    #     ratio_range=(0.1, 2.0),
    #     keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_type='absolute_range',
    #     crop_size=image_size,
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    # dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=5,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_kfold_0.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline
        ))

val_dataloader = dict(
    batch_size=1,
    num_workers=5,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val_kfold_0.json',
        data_prefix=dict(img=''),
        ))

# test 시 batchsize를 변경하시면 됩니다.
test_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.json',
        data_prefix=dict(img=''),
        ))


#### Evaluation ####
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_kfold_0.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    )

test_evaluator = dict(ann_file=data_root + 'test.json')

### Optimizer ###
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.05)}))


# learning policy
max_epochs = 6
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

### Scheduler 세팅 ###
# 6 epoch을 기준으로 작성됨 MultiStep, CosineAnnealing 중 하나 선택
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=False,
        begin=0,
        end=2000),
    dict(
        type='MultiStepLR',
        begin=1,
        end=max_epochs,
        by_epoch=True,
        milestones=[4, 5],
        gamma=0.1),
    # dict(
    #     type='CosineAnnealingLR',
    #     eta_min=0.0,
    #     begin=1,
    #     T_max=max_epochs,
    #     end=max_epochs,
    #     by_epoch=True,
    #     convert_to_iter_based=True)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
# 우리는 GPU가 1개이므로, base_batch_size = 2
auto_scale_lr = dict(base_batch_size=2)

#### Wandb ####
# name: 자신이 추가한 내용을 키워드로 간단하게 적어주시면 됩니다.
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'mmdetection',
            'entity': 'ai_tech_level2_objectdetection',
            'group': 'ddq',
            'name': 'default'
         })]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

#### Pretrained Model ####
load_from = "https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_swinl_30e.pth" 
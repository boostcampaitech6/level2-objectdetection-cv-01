_base_ = [
    '../configs/yolox/yolox_l_8xb8-300e_coco.py'
] # 모델 config 파일 주소를 상대주소로 가져옴(지금 파일 기준!!!!!! 무조건!!!!! 안그러면 에러남)


# model 끝단에 num_classes 부분을 바꿔주기 위해 해당 모듈을 불러와 선언해줍니다.
model = dict(
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0))
    )

# custom hooks을 만들었다면 여기서 선언해 사용할 수 있습니다.
custom_hooks = [
    dict(type='SubmissionHook')
]

metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}

# dataset 설정을 해줍니다.
data_root = '/data/ephemeral/home/dataset/'


train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    dataset=dict(
        data_root=data_root,
        ann_file='train_kfold_0.json',
        metainfo=metainfo,
        data_prefix=dict(img=''),
        )
    )



train_dataloader = dict(
    batch_size=4,
    num_workers=5,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)



val_dataloader = dict(
    batch_size=1,
    num_workers=5,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val_kfold_0.json',
        data_prefix=dict(img='')
        ))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.json',
        data_prefix=dict(img='')))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_kfold_0.json',
    metric='bbox',
    )

test_evaluator = dict(ann_file=data_root + 'test.json')

# training settings
max_epochs = 100
num_last_epochs = 15
interval = 1

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=10  # only keep latest 3 checkpoints
    ))

load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth" 
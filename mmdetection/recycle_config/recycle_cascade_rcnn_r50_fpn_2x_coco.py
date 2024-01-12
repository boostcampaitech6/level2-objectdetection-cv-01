_base_ = [
    '../configs/_base_/models/cascade-rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_detection.py',
    '../configs/_base_/schedules/schedule_2x.py',
    '../configs/_base_/default_runtime.py'
]
'''
변경 사항


'''

# Model
for i in range(3): _base_.model.roi_head.bbox_head[i].num_classes = 10

# Custom Hook
custom_hooks = [
    dict(type='SubmissionHook')
]

# Scheduling
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_begin=12, val_interval=1)

# Wandb
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'mmdetection',
            'entity': 'ai_tech_level2_objectdetection',
            'group': 'cascade_rcnn',
            'name': 'full_dataset_2x'
         })
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# Dataset
data_root = '/data/ephemeral/home/dataset/'
metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}

albu_train_transforms = [
    dict(
        type='RandomSizedBBoxSafeCrop',
        height=1024,
        width=1024,
        erosion_rate=0.3,
        interpolation=1,
        p=0.3
        ),
    dict(
        type='OneOf',
        transforms=
        [
            dict(type='VerticalFlip', p=1),
            dict(type='HorizontalFlip', p=1),
        ], p=0.5
        ),
    dict(
        type='ToGray',
        p=0.1
        ),
    dict(
        type='GaussNoise',
        var_limit=(20, 100), 
        p = 0.3
        ),
    dict(
        type='OneOf',
        transforms=
        [
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='GaussianBlur', p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0),
            dict(type='MotionBlur', p = 1.0)
        ],
        p=0.2
        ),
    dict(
        type='CLAHE',
        p=0.3
        ),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=(0.0, 0.15),
        contrast_limit=(0.0, 0.15),
        p=0.3
        ),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=10,
        p=0.3
        )
]

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

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='Albu', transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_labels'],
    #         min_visibility=0.0,
    #         filter_lost_elements=True
    #     ),
    #     keymap={
    #         'img': 'image',
    #         'gt_masks': 'masks',
    #         'gt_bboxes': 'bboxes'
    #     },
    #     update_pad_shape=False, skip_img_without_anno=True),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator

load_from = "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth" 
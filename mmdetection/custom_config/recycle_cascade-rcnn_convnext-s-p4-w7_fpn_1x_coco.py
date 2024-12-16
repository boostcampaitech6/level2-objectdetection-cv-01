_base_ = [
    '../configs/_base_/models/cascade-rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_detection_recycle.py',
    '../configs/_base_/schedules/schedule_1x.py', '../configs/_base_/default_runtime.py' 
]

# classification에 대한 class수 할당
for i in range(len(_base_.model.roi_head.bbox_head)):
    _base_.model.roi_head.bbox_head[i].num_classes = 10

# model laad
custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'  # noqa


# backbone & neck 변경 
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='small',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.6,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
        neck=dict(in_channels=[96, 192, 384, 768], start_level=0))

# test.py 시에 사용하는 SubmissionHook
custom_hooks = [
    dict(type='SubmissionHook')
]

# wandb 등록 : name에 실험이름 설정 
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'mmdetection',
            'entity': 'ai_tech_level2_objectdetection',
            'group': 'cascade_rcnn',
            'name' : 'default'
         })
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')


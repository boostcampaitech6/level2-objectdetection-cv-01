_base_ = [
    '../configs/_base_/models/cascade-rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_detection_recycle.py',
    '../configs/_base_/schedules/schedule_1x.py', '../configs/_base_/default_runtime.py'
]

# classification에 대한 class수 할당
for i in range(len(_base_.model.roi_head.bbox_head)):
    _base_.model.roi_head.bbox_head[i].num_classes = 10

# model laad
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

# backbone & neck 변경 
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))

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

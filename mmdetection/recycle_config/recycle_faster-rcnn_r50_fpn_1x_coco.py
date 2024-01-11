_base_ = [
    '../configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
]


custom_hooks = [
    dict(type='SubmissionHook')
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'mmdetection',
            'entity': 'ai_tech_level2_objectdetection',
            'group': 'cascade_rcnn',
            'name': 'exp'
         })]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# pretrained 모델 관련
load_from = "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth"


_base_ = [
    './cascade_rcnn_r50_fpn.py',
    './coco_detection.py',
    './schedule_1x.py', './default_runtime.py'
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

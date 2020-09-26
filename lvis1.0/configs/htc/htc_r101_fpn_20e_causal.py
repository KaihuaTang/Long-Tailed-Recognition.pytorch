_base_ = './htc_without_semantic_r50_fpn_1x_lvis.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
# learning policy
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[16, 19])
total_epochs = 20

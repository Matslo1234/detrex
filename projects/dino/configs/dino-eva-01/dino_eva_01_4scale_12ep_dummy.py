from .dino_eva_01_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)


model.backbone.square_pad = 640
model.backbone.net.img_size = 640
dataloader.train.dataset.names = "uxo_11_resized_train"
dataloader.test.dataset.names = "uxo_11_resized_valid"
# modify dataloader config
dataloader.train.num_workers = 1

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 1

model.num_classes = 11

train.eval_period = 1000 # TODO change
train.init_checkpoint = "/home/m/Downloads/dino_eva_01_o365_finetune_detr_like_augmentation_4scale_12ep.pth"
from .dino_r50_4scale_24ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# actually works :)

dataloader.train.dataset.names = "uxo_11_cropped_256_1class_train"
dataloader.test.dataset.names = "uxo_11_cropped_256_1class_valid"
# modify dataloader config
dataloader.train.num_workers = 1

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 1

model.num_classes = 11

train.amp.enabled = False
train.eval_period = 5000
train.init_checkpoint = "/home/m/Downloads/dino_r50_4scale_24ep.pth"
train.output_dir = "./output/dino_r50_4scale_24ep_dummy_cropped"

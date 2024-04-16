from .dino_r50_4scale_12ep_300dn import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

# actually works :)

dataloader.train.dataset.names = "uxo_11_fair_1class_train"
dataloader.test.dataset.names = "uxo_11_fair_1class_valid"
# modify dataloader config
dataloader.train.num_workers = 1

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 1

model.num_classes = 1

train.amp.enabled = False
train.eval_period = 5000
train.init_checkpoint = "/home/m/Downloads/dino_r50_4scale_12ep_backbone_2e-5_class_weight_2.0_300dn.pth"
train.output_dir = "./output/dino_r50_4scale_12ep_300dn_fair_1class"

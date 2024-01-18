from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)
# actually works :)

dataloader.train.dataset.names = "uxo_11_train"
dataloader.test.dataset.names = "uxo_11_valid"
# modify dataloader config
dataloader.train.num_workers = 1

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 1

model.num_classes = 11

train.eval_period = 5000
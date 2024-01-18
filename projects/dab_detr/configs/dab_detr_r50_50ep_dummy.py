from .dab_detr_r50_50ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
    model,
)

dataloader.train.dataset.names = "uxo_11_train"
dataloader.test.dataset.names = "uxo_11_valid"
# modify dataloader config
dataloader.train.num_workers = 2

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 2

model.num_classes = 11

train.eval_period = 500

from detrex.config import get_config
from ..models.dino_r50_custom import model

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# max training iterations
train.max_iter = 90000
train.eval_period = 5000
train.log_period = 20
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# dump the testing results into output_dir for visualization

# actually works :)
dataloader.train.mapper.img_format = "grayscale"
dataloader.train.dataset.names = "uxo_11_cropped_256_1class_train"
dataloader.test.dataset.names = "uxo_11_cropped_256_1class_valid"
dataloader.test.mapper.img_format = "grayscale"

# modify dataloader config
dataloader.train.num_workers = 1

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 1

model.num_classes = 1
model.pixel_mean = [103.530]
model.pixel_std = [57.375]

train.amp.enabled = False
train.eval_period = 5000
train.output_dir = "./output/dino_r50_custom_cropped256/"
dataloader.evaluator.output_dir = train.output_dir

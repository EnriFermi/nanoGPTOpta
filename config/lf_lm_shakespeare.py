# config/lf_lm_shakespeare.py

out_dir = 'out-lf-adamlm'
dataset = 'shakespeare_char'

eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = True
wandb_project = 'shakespeare-char'
wandb_run_name = 'lf-adamlm'

gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 3e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

optimizer_name = 'lowfreq_adam_multi'   # uses LowFreqAdamLM under the hood
optimizer_kwargs = {
    "specs": {
        "block0": {"m": 16, "sigma": 0.8, "alpha": 1.0},
        "block1": {"m": 16, "sigma": 0.8, "alpha": 1.0},
        "block2": {"m": 16, "sigma": 0.8, "alpha": 1.0},
        "block3": {"m": 16, "sigma": 0.8, "alpha": 1.0},
        "block4": {"m": 16, "sigma": 0.8, "alpha": 1.0},
        "block5": {"m": 16, "sigma": 0.8, "alpha": 1.0},
    },
    "lam": 0.5,
    "scale_match": True,
    "N_images": 0,
}

# LowFreqAdamLM config for Shakespeare char (6-layer small model).

out_dir = 'out-lf-adam-lm'
dataset = 'shakespeare_char'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False
wandb_log = True
wandb_project = 'shakespeare-char'
wandb_run_name = 'lf-adam-multi'

gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

optimizer_name = 'lowfreq_adam_multi'
lowfreq_m = 16
lowfreq_sigma = 0.8
lowfreq_lam = 0.5
lowfreq_scale_match = True

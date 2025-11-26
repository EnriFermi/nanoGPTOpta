SEED=1337

# 1) Baseline Adam (overparameterized)
torchrun --standalone --nproc_per_node=1 train.py \
  --dataset=shakespeare_char \
  --out_dir=out-adam-small \
  --eval_interval=250 --eval_iters=200 --log_interval=10 \
  --always_save_checkpoint=False \
  --wandb_log=True --wandb_project=shakespeare-char --wandb_run_name=adam-small \
  --gradient_accumulation_steps=1 --batch_size=64 --block_size=256 \
  --n_layer=6 --n_head=6 --n_embd=384 --dropout=0.2 \
  --learning_rate=1e-3 --max_iters=5000 --lr_decay_iters=5000 --min_lr=1e-4 \
  --beta2=0.99 --warmup_iters=100 \
  --optimizer_name=adam \
  --device=cuda --dtype=bfloat16 --compile=False \
  --seed=$SEED

# 2) Low-frequency Adam (same overparameterized model)
torchrun --standalone --nproc_per_node=1 train.py \
  --dataset=shakespeare_char \
  --out_dir=out-lf-adam-small \
  --eval_interval=250 --eval_iters=200 --log_interval=10 \
  --always_save_checkpoint=False \
  --wandb_log=True --wandb_project=shakespeare-char --wandb_run_name=lf-adam-small \
  --gradient_accumulation_steps=1 --batch_size=64 --block_size=256 \
  --n_layer=6 --n_head=6 --n_embd=384 --dropout=0.2 \
  --learning_rate=1e-3 --max_iters=5000 --lr_decay_iters=5000 --min_lr=1e-4 \
  --beta2=0.99 --warmup_iters=100 \
  --optimizer_name=lowfreq_adam \
  --lowfreq_m=16 --lowfreq_sigma=0.8 --lowfreq_lam=0.5 --lowfreq_scale_match=True \
  --device=cuda --dtype=bfloat16 --compile=False \
  --seed=$SEED




torchrun --standalone --nproc_per_node=1 train.py \
  --dataset=shakespeare_char \
  --out_dir=out-adamcustom-small \
  --eval_interval=250 --eval_iters=200 --log_interval=10 \
  --always_save_checkpoint=False \
  --wandb_log=True --wandb_project=shakespeare-char --wandb_run_name=adamcustom-small \
  --gradient_accumulation_steps=1 --batch_size=64 --block_size=256 \
  --n_layer=6 --n_head=6 --n_embd=384 --dropout=0.2 \
  --learning_rate=1e-3 --max_iters=5000 --lr_decay_iters=5000 --min_lr=1e-4 \
  --beta2=0.99 --warmup_iters=100 \
  --optimizer_name=adam_custom \
  --device=cuda --dtype=bfloat16 --compile=False \
  --seed=$SEED



torchrun --standalone --nproc_per_node=1 train.py \
  --dataset=shakespeare_char \
  --out_dir=out-adamcustom-small-nosqrt \
  --eval_interval=250 --eval_iters=200 --log_interval=10 \
  --always_save_checkpoint=False \
  --wandb_log=True --wandb_project=shakespeare-char --wandb_run_name=adamcustom-small-nosqrt \
  --gradient_accumulation_steps=1 --batch_size=64 --block_size=256 \
  --n_layer=6 --n_head=6 --n_embd=384 --dropout=0.2 \
  --learning_rate=1e-3 --max_iters=5000 --lr_decay_iters=5000 --min_lr=1e-4 \
  --beta2=0.99 --warmup_iters=100 \
  --optimizer_name=adam_custom_nosqrt \
  --device=cuda --dtype=bfloat16 --compile=False \
  --seed=$SEED
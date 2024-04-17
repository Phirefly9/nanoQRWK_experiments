# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'rwkv-shakespeare-char'
wandb_run_name = 'gpt2-baseline'

dataset = 'shakespeare_char'

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 1000
lr_decay_iters = 600000

# eval stuff
eval_interval = 50
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
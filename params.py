import os

gpu_avilibale = True
if not gpu_avilibale:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
frames_dir = "/dsi/gannot-lab2/datasets2/FAIR-Play/frames_30fps/"
audios_dir = "/dsi/gannot-lab2/datasets2/FAIR-Play/binaural_audios/"
batch_size = 64
epochs = 1000
gpu_ids = [0,1,2,3]
lr = 1e-4
lr_big = 1e-3 
beta1 = 0.9
weight_decay = 0.0005 # use for regolization
train_epochs = 1000
checkpoints_dir = "/dsi/bermanl1/CODE/checkpoints"
learning_rate_decrease_itr = 10
decay_factor = 0.94
alpha = 0

display_freq = 50     #display_freq batches the training progress 
save_epoch_freq = 50
save_latest_freq = 5000
validation_freq = 100

# weights of loss
lambda_b = 10
lambda_s = 1
lambda_g = 0.01
lambda_p = 1
lambda_f = 1
lambda_binarual =1
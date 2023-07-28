import os
import torch

gpu_ids = [5]
gpu_available = True
devices = []

if not gpu_available:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
if gpu_available and torch.cuda.is_available():
    if len(gpu_ids) == 1:
        device = torch.device('cuda', gpu_ids[0])
        devices.append(device)
    elif len(gpu_ids) > 1:
        for i in gpu_ids:
            device = torch.device('cuda', i)
            devices.append(device)
else:
    device = torch.device('cpu')
    
frames_dir = "/dsi/gannot-lab2/datasets2/FAIR-Play/frames_30fps/"
audios_dir = "/dsi/gannot-lab2/datasets2/FAIR-Play/binaural_audios/"
debug_dir = "/home/dsi/bermanl1/Geometry-Aware-Multi-Task-Learning-for-Binaural-Audio-Generation-from-Video/pic_for_debug"
batch_size = 64
batch_size_test = 1
epochs = 1000


lr = 1e-4
lr_big = 7e-4 

beta1 = 0.9
weight_decay = 0.0005 # use for regolization
train_epochs = 1000
checkpoints_dir = "/home/dsi/bermanl1/CODE/checkpoints/"
learning_rate_decrease_itr = 10
decay_factor = 0.94
alpha = 0

display_freq = 50     #display_freq batches the training progress 
save_epoch_freq = 50
save_latest_freq = 5000
validation_freq = 100
test_overlap = 0.5

# weights of loss
lambda_b = 10
lambda_s = 1
lambda_g = 0.01
lambda_p = 1
lambda_f = 1
lambda_binarual =1

audio_length = 0.63
audio_sampling_rate = 16000
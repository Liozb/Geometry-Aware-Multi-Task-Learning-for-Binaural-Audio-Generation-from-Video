# this code handles the train

from Models.backbone_model import *
from imports import * 
from Datasets.AudioVisualDataset import AudioVisualDataset
from networks.Networks import *


def debug_dataset(dataset, idx=15):
    """ debug function to whatch a specific index in the dataset.
        saves the output to the dubug folder. 

    Args:
        idx (int, optional): index for a place in the dataset. Defaults to 15.
    """
    data_idx = dataset[idx]
    frame_idx = data_idx['frame']
    audio_spec_idx = data_idx['audio_diff_spec']

    audio_spec_idx = torch.sqrt(audio_spec_idx[0,:,:]**2 + audio_spec_idx[1,:,:]**2) 
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audio_spec_idx), ref=np.max),
                            y_axis='log', x_axis='time', cmap='bwr')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB)')
    plt.savefig('pic_for_debug/audio_spec.jpg', format='jpg')

    plt.imshow(frame_idx.permute(1,2,0).numpy())
    plt.savefig('pic_for_debug/frame.jpg', format='jpg')
    
    

frames_dir = "/dsi/gannot-lab/datasets2/FAIR-Play/frames_30fps/"
audios_dir = "/dsi/gannot-lab/datasets2/FAIR-Play/binaural_audios/"
batch_size = 64
gpu_ids = [0]
lr = 1e-4
backbone_lr = 1e-3 
beta1 = 0.9
weight_decay = 0.0005 # use for regolization
train_epochs = 1000
display_freq = 50     # display every #display_freq batches the training progress 
validation_freq = 100
checkpoints_dir = "/dsi/bermanl1/CODE/checkpoints"
learning_rate_decrease_itr = 10
decay_factor = 0.94
save_epoch_freq = 50


dataset = AudioVisualDataset(audios_dir, frames_dir)
data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(dataset.nThreads))


# validation dataset
dataset.mode = 'val'
val_dataset = AudioVisualDataset(audios_dir, frames_dir)
data_loader_val = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(val_dataset.nThreads))
dataset.mode = 'train'

# call the frames data from the folders
resnet18 = models.resnet18(pretrained=True)
visual_net = VisualNet(resnet18)

audio_net = AudioNet()
audio_net.apply(weights_init)

# construct our models
model_backbone = BackboneModel(audio_net)


# use models with gpu
model_backbone = torch.nn.DataParallel(model_backbone, device_ids=gpu_ids)
model_backbone.to(dataset.device)

#define Adam optimzer
param_backbone = [{'params': visual_net.parameters(), 'lr': backbone_lr},
                {'params': audio_net.parameters(), 'lr': backbone_lr}]
optimizer_backbone = torch.optim.Adam(param_backbone, betas=(beta1,0.999), weight_decay=weight_decay)

# set up loss function
loss_criterion = torch.nn.MSELoss()


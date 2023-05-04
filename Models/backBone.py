import sys
import os
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
from imports import *
from Datasets.AudioVisualDataset import AudioVisualDataset
from networks.unet import unet


def debug_dataset(idx=15):
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

# define device as gpu
dataset = AudioVisualDataset(audios_dir, frames_dir)
data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(dataset.nThreads))


# validation dataset
val_dataset = AudioVisualDataset(audios_dir, frames_dir)
data_loader_val = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(val_dataset.nThreads))

# debug_dataset()

# call the frames data from the folders
resnet18 = models.resnet18(pretrained=True)

#  remove the pooling and fully connected layers
modules = list(resnet18.children())[:-2]
visual_net = nn.Sequential(*modules)

audio_net = AudioNet()
audio_net.apply(weights_init)

resnet18.eval()
resnet18.to(dataset.device)

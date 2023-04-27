from imports import *
from Datasets.AudioVisualDataset import AudioVisualDataset


frames_dir = "/dsi/gannot-lab/datasets2/FAIR-Play/frames_30fps/"
audios_dir = "/dsi/gannot-lab/datasets2/FAIR-Play/binaural_audios/"
batch_size = 32

# define device as gpu
dataset = AudioVisualDataset(audios_dir, frames_dir)
data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(dataset.nThreads))
data_15 = dataset[15]
frame_15 = data_15['frame']
audio_spec_15 = data_15['audio_diff_spec']


librosa.display.specshow(audio_spec_15.numpy(), sr=dataset.audio_sampling_rate, x_axis='time', y_axis='hz')
plt.colorbar()
plt.show()


plt.imshow(frame_15.permute(1,2,0).numpy())
plt.show()

# validation dataset
val_dataset = AudioVisualDataset(audios_dir, frames_dir)
data_loader_val = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(val_dataset.nThreads))

# call the frames data from the folders
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
resnet18.to(dataset.device)

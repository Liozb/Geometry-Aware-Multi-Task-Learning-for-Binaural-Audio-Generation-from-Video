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
print(dataset[15])


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

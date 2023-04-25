from imports import *
from Datasets.AudioVisualDataset import AudioVisualDataset


frames_dir = "/dsi/gannot-lab/datasets2/FAIR-Play/frames_30fps/"
audios_dir = "/dsi/gannot-lab/datasets2/FAIR-Play/binaural_audios/"
batch_size = 32

# define device as gpu
fair_play_dataset = AudioVisualDataset(audios_dir, frames_dir)
fair_play_dataset.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_loader = CustomDatasetDataLoader(fair_play_dataset)
dataset = data_loader.load_data()

# validation dataset
data_loader_val = AudioVisualDataset(audios_dir, frames_dir)
data_loader_val.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_loader_val = CustomDatasetDataLoader(data_loader_val)
dataset_val = data_loader_val.load_data()


# call the frames data from the folders
resnet18 = models.resnet18(pretrained=True)


# Create a DataLoader to load the audio files in batches
audio_dataloader = DataLoader(audio_dataset, batch_size=32, shuffle=True)


resnet18.eval()

resnet18.to(data_loader.device)

# code start here
with torch.no_grad():
    for images, _ in frames_loader:
        images = images.to(data_loader.device)
        outputs = resnet18(images)
        _, predicted = torch.max(outputs, 1)
        print(predicted)
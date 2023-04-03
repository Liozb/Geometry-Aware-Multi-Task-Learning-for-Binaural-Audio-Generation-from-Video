import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize
import torchvision.datasets as datasets
import torchvision.models as models
import os


class AudioVisualDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(AudioVisualDataset, self)
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []

        # Find all the audio files in the root directory and subdirectories
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.wav') or file.endswith('.mp3'):
                    self.file_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load the audio file and its corresponding label
        file_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        # Apply any transforms to the waveform
        if self.transform:
            waveform = self.transform(waveform)

        return waveform


# call the frames data from the folders
resnet18 = models.resnet18(pretrained=True)
frames_path = "/dsi/gannot-lab/datasets2/FAIR-Play/frames_30fps/"
batch_size = 64

# load visual dataset
frame_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

frames_dataset = datasets.ImageFolder(root=frames_path, transform=frame_transform)
frames_loader = DataLoader(frames_dataset, batch_size=32, shuffle=True)

# load audio dataset
audio_transform = Compose([
    torchaudio.transforms.Resample(orig_freq=44100, new_freq=8000),
    torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_fft=1024, hop_length=256),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35),
    Normalize(mean=0.5, std=0.5)
])

# Load the audio dataset from the root directory
audio_dataset = AudioVisualDataset(root_dir='path/to/root/directory', transform=audio_transform)

# Create a DataLoader to load the audio files in batches
audio_dataloader = DataLoader(audio_dataset, batch_size=32, shuffle=True)

# define device as gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet18.eval()

resnet18.to(device)

# code start here
with torch.no_grad():
    for images, _ in frames_loader:
        images = images.to(device)
        outputs = resnet18(images)
        _, predicted = torch.max(outputs, 1)
        print(predicted)

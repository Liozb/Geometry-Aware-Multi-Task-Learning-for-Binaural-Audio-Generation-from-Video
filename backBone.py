import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchvision.transforms as transforms
import torchvision.datasets as datasets


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root='/dsi/gannot-lab/datasets2/FAIR-Play/frames_30fps/', transform=transform)


dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for images, labels in dataloader:
    print(images)
    # Do something with the batch of images and labels

class FairPlayAudio(Dataset):
    def __init__(self, audio, labels):
        self.labels = labels
        self.audio = audio

    def __getitem__(self, idx):
        label = self.labels[idx]
        audio = self.audio[idx]
        sample = {"audio": audio, "Class": label}
        return sample

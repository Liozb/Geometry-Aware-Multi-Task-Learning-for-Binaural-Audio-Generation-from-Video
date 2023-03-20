import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)


frames_path = "/dsi/gannot-lab/datasets2/FAIR-Play/frames_30fps/"
batch_size = 64

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=frames_path, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet18.eval()
resnet18.to(device)

with torch.no_grad():
    for images, _ in loader:
        images = images.to(device)
        outputs = resnet18(images)
        _, predicted = torch.max(outputs, 1)
        print(predicted)

class FairPlayAudio(Dataset):
    def __init__(self, audio, labels):
        self.labels = labels
        self.audio = audio

    def __getitem__(self, idx):
        label = self.labels[idx]
        audio = self.audio[idx]
        sample = {"audio": audio, "Class": label}
        return sample

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
import librosa
import random
from PIL import Image, ImageEnhance


def normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples ** 2)))
    samples = samples * (desired_rms / rms)
    return samples


def process_image(image, augment):
    image = image.resize((480, 240))
    w, h = image.size
    w_offset = w - 448
    h_offset = h - 224
    left = random.randrange(0, w_offset + 1)
    upper = random.randrange(0, h_offset + 1)
    image = image.crop((left, upper, left+448, upper+224))
    if augment:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.random() * 0.6 + 0.7)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.random() * 0.6 + 0.7)
    return image


def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel


class FairPlayDataset(Dataset):
    def __init__(self, audio_dir, frame_dir):
        super(FairPlayDataset, self).__init__()
        self.audio_dir = audio_dir
        self.frame_dir = frame_dir
        self.audio_length = 0.63            # the audio for each length is 0.63 sec
        self.audio_sampling_rate = 16000    # sampling rate for each audio
        self.batch_size = 32
        self.enable_data_augmentation = True
        self.audios = []

        for file_name in os.listdir(audio_dir):
            file_path = os.path.join(audio_dir, file_name)
            # Check if the path points to a file (as opposed to a directory)
            if os.path.isfile(file_path):
                self.audios.append(file_path)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, index):
        # load audio
        audio, audio_rate = librosa.load(self.audios[index], sr=self.audio_sampling_rate, mono=False)

        # randomly get a start time for the audio segment from the 10s clip
        audio_start_time = random.uniform(0, 9.9 - self.audio_length)
        audio_end_time = audio_start_time + self.audio_length
        audio_start = int(audio_start_time * self.audio_sampling_rate)
        audio_end = audio_start + int(self.audio_length * self.audio_sampling_rate)
        audio = audio[:, audio_start:audio_end]
        audio = normalize(audio)
        audio_channel1 = audio[0, :]
        audio_channel2 = audio[1, :]

        # get the frame dir path based on audio path
        path_parts = self.audios[index].strip().split('/')
        video_num = path_parts[-1][:-4]

        # get the closest frame to the audio segment
        frame_index = int(
            round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10))  # 10 frames extracted per second
        frame = process_image(Image.open(os.path.join(self.frame_dir, video_num, str(frame_index) + '.jpg')).convert('RGB'),
                              self.enable_data_augmentation)
        frame = self.vision_transform(frame)

        #passing the spectrogram of the difference
        audio_diff_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 - audio_channel2))
        audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 + audio_channel2))

        return {'frame': frame, 'audio_diff_spec':audio_diff_spec, 'audio_mix_spec':audio_mix_spec}


# call the frames data from the folders
resnet18 = models.resnet18(pretrained=True)
frames_dir = "/dsi/gannot-lab/datasets2/FAIR-Play/frames_30fps/"
audios_dir = "/dsi/gannot-lab/datasets2/FAIR-Play/binaural_audios/"                 # check for real path


frames_dataset = datasets.ImageFolder(root=frames_path, transform=frame_transform)
frames_loader = DataLoader(frames_dataset, batch_size=32, shuffle=True)

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

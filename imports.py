import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize
import torchvision.datasets as datasets
import torchvision.models as models
import os
import librosa
import random
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
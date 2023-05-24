# this file responsible for the spatial coherence train model

import sys
import os
from Models.backbone_model import *
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
from imports import *
from Datasets.AudioVisualDataset import AudioVisualDataset

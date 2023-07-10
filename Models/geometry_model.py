# this file is not in a use

import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
from params import *

class modelGeometry(torch.nn.Module):
    def __init__(self, geometric_visual):
        super(modelGeometry, self).__init__()
        self.geometric_visual = geometric_visual
        
    
    def forward(self, data):
        second_visual_input = data['second_frame'].to(device)
        second_visual_feature = self.geometric_visual.forward(second_visual_input)
        return second_visual_feature 
    
if __name__ == "__main__":
    x = torch.randn(1, )
    net = modelGeometry()
    
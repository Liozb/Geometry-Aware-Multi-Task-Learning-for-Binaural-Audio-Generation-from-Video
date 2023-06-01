import sys
import os
from Models.backbone_model import *
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
from imports import *


class modelRir(torch.nn.Module):
    def __init__(self, gen_net):
        super(modelRir, self).__init__()
        self.gen_net = gen_net
        

    def forward(self, visual_feature):
        pred_rir = self.gen_net(visual_feature)
        
    

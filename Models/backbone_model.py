import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable


class modelBackbone(torch.nn.Module):
    def __init__(self, audio_net):
        super(modelBackbone, self).__init__()
        #initialize model
        self.net_audio = audio_net
        self.name = "backbone"

    def forward(self, input, visual_feature, volatile=False):
        audio_diff = input['audio_diff_spec']
        audio_mix = input['audio_mix_spec']
        audio_gt = Variable(audio_diff[:,:,:-1,:], requires_grad=False)  # discarding the last time frame of the spectrogram(why?)

        input_spectrogram = Variable(audio_mix, requires_grad=False, volatile=volatile)
        mask_prediction = self.net_audio(input_spectrogram, visual_feature, self.name)

        #complex masking to obtain the predicted spectrogram
        spectrogram_diff_real = input_spectrogram[:,0,:-1,:] * mask_prediction[:,0,:,:] - input_spectrogram[:,1,:-1,:] * mask_prediction[:,1,:,:]
        spectrogram_diff_img = input_spectrogram[:,0,:-1,:] * mask_prediction[:,1,:,:] + input_spectrogram[:,1,:-1,:] * mask_prediction[:,0,:,:]
        binaural_spectrogram = torch.cat((spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1)
        
        # predicted channels
        channel1_pred = input_spectrogram + (input_spectrogram*mask_prediction)/2
        channel2_pred = input_spectrogram - (input_spectrogram*mask_prediction)/2

        output =  {'mask_prediction': mask_prediction, 'binaural_spectrogram': binaural_spectrogram, 'audio_gt': audio_gt, 'channel1_pred': channel1_pred, 'channel2_pred': channel2_pred}
        return output
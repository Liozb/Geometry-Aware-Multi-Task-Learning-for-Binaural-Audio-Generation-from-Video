import torch

"""
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
"""

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()








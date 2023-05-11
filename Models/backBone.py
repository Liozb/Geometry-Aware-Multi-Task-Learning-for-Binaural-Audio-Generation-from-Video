import sys
import os
from audioVisual_model import *
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR))
from imports import *
from Datasets.AudioVisualDataset import AudioVisualDataset
from networks.networks import *


#used to display validation loss
def display_val(model, loss_criterion, writer, index, dataset_val, validation_batches=10):
    losses = []
    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            if i < validation_batches:
                output = model.forward(val_data)
                loss = loss_criterion(output['binaural_spectrogram'], output['audio_gt'])
                losses.append(loss.item()) 
            else:
                break
    avg_loss = sum(losses)/len(losses)
    writer.add_scalar('data/val_loss', avg_loss, index)
    print('val loss: %.3f' % avg_loss)
    return avg_loss 


def debug_dataset(idx=15):
    """ debug function to whatch a specific index in the dataset.
        saves the output to the dubug folder. 

    Args:
        idx (int, optional): index for a place in the dataset. Defaults to 15.
    """
    data_idx = dataset[idx]
    frame_idx = data_idx['frame']
    audio_spec_idx = data_idx['audio_diff_spec']

    audio_spec_idx = torch.sqrt(audio_spec_idx[0,:,:]**2 + audio_spec_idx[1,:,:]**2) 
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audio_spec_idx), ref=np.max),
                            y_axis='log', x_axis='time', cmap='bwr')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB)')
    plt.savefig('pic_for_debug/audio_spec.jpg', format='jpg')



    plt.imshow(frame_idx.permute(1,2,0).numpy())
    plt.savefig('pic_for_debug/frame.jpg', format='jpg')
    
name = 'Backbone'
frames_dir = "/dsi/gannot-lab/datasets2/FAIR-Play/frames_30fps/"
audios_dir = "/dsi/gannot-lab/datasets2/FAIR-Play/binaural_audios/"
batch_size = 64
gpu_ids = []
lr = 1e-3
beta1 = 0.9
weight_decay = 0.0005 # use for regolization
train_epochs = 1000
display_freq = 50     # display every #display_freq batches the training progress 
validation_freq = 100
checkpoints_dir = "/dsi/bermanl1/CODE/checkpoints"
learning_rate_decrease_itr = 10
decay_factor = 0.94
save_epoch_freq = 50

# define device as gpu
dataset = AudioVisualDataset(audios_dir, frames_dir)
data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(dataset.nThreads))


# validation dataset
dataset.mode = 'val'
val_dataset = AudioVisualDataset(audios_dir, frames_dir)
data_loader_val = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(val_dataset.nThreads))
dataset.mode = 'train'

# debug_dataset()

from tensorboardX import SummaryWriter
writer = SummaryWriter(comment=opt.name)

# call the frames data from the folders
resnet18 = models.resnet18(pretrained=True)
visual_net = VisualNet(resnet18)

audio_net = AudioNet()
audio_net.apply(weights_init)

nets = (visual_net,audio_net)

# construct our audio-visual model
model = AudioVisualModel(nets)
if len(gpu_ids) > 0:
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
else:
    model = torch.nn.DataParallel(model)

model.to(dataset.device)


param_groups = [{'params': visual_net.parameters(), 'lr': lr},
                {'params': audio_net.parameters(), 'lr': lr}]
optimizer = torch.optim.Adam(param_groups, betas=(beta1,0.999), weight_decay=weight_decay)

# set up loss function
loss_criterion = torch.nn.MSELoss()
if(len(gpu_ids) > 0):
    loss_criterion.cuda(gpu_ids[0])

# initialization
total_steps = 0
data_loading_time = []
model_forward_time = []
model_backward_time = []
batch_loss = []
best_err = float("inf")

for epoch in range(1, train_epochs):
    torch.cuda.synchronize()
                
    for i, data in enumerate(data_loader):
        
        total_steps += batch_size

        # forward pass
        model.zero_grad()
        output = model.forward(data)

        # compute loss
        difference_loss = loss_criterion(output['binaural_spectrogram'], Variable(output['audio_gt'], requires_grad=False))
        channel1_loss = loss_criterion(output['channel1_pred'], data['channel1_spec'])
        channel2_loss = loss_criterion(output['channel2_pred'], data['channel2_spec'])
        loss = difference_loss + channel1_loss + channel2_loss
        batch_loss.append(loss.item())

        # update optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(total_steps // batch_size % display_freq == 0):
                print('Display training progress at (epoch %d, total_steps %d)' % (epoch, total_steps))
                avg_loss = sum(batch_loss) / len(batch_loss)
                print('Average loss: %.3f' % (avg_loss))
                batch_loss = []
                writer.add_scalar('data/loss', avg_loss, total_steps)
                print('end of display \n')

        if(total_steps // batch_size % validation_freq == 0):
                model.eval()
                dataset.mode = 'val'
                print('Display validation results at (epoch %d, total_steps %d)' % (epoch, total_steps))
                val_err = display_val(model, loss_criterion, writer, total_steps, data_loader_val)
                print('end of display \n')
                model.train()
                dataset.mode = 'train'
                #save the model that achieves the smallest validation error
                if val_err < best_err:
                    best_err = val_err
                    print('saving the best model (epoch %d, total_steps %d) with validation error %.3f\n' % (epoch, total_steps, val_err))
                    torch.save(visual_net.state_dict(), os.path.join('.', checkpoints_dir, name, 'visual_best.pth'))
                    torch.save(audio_net.state_dict(), os.path.join('.', checkpoints_dir, name, 'audio_best.pth'))        


    if(epoch % save_epoch_freq == 0):
        print('saving the model at the end of epoch %d, total_steps %d' % (epoch, total_steps))
        torch.save(visual_net.state_dict(), os.path.join('.', checkpoints_dir, name, str(epoch) + '_visual.pth'))
        torch.save(audio_net.state_dict(), os.path.join('.', checkpoints_dir, name, str(epoch) + '_audio.pth'))

    #decrease learning rate 6% every opt.learning_rate_decrease_itr epochs
    if(learning_rate_decrease_itr > 0 and epoch % learning_rate_decrease_itr == 0):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_factor
        print('decreased learning rate by ', decay_factor)
    
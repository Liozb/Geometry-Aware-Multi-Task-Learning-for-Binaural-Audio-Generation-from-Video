# this code handles the train

from Models.backbone_model import *
from imports import * 
from Datasets.AudioVisualDataset import AudioVisualDataset
from networks.Networks import *


def debug_dataset(dataset, idx=15):
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
    
def display_val(model, loss_criterion, writer, index, dataset_val):
    # number of batches to test for validation
    val_batch = 10 
    
    losses = []
    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            if i < val_batch:
                output = model.forward(val_data)
                loss = loss_criterion(output['binaural_spectrogram'], output['audio_gt'])
                losses.append(loss.item()) 
            else:
                break
    avg_loss = sum(losses)/len(losses)
    writer.add_scalar('data/val_loss', avg_loss, index)
    print('val loss: %.3f' % avg_loss)
    
    return avg_loss 
    
    

frames_dir = "/dsi/gannot-lab/datasets2/FAIR-Play/frames_30fps/"
audios_dir = "/dsi/gannot-lab/datasets2/FAIR-Play/binaural_audios/"
batch_size = 64
epochs = 1000
gpu_ids = [0,1]
lr = 1e-4
backbone_lr = 1e-3 
beta1 = 0.9
weight_decay = 0.0005 # use for regolization
train_epochs = 1000
checkpoints_dir = "/dsi/bermanl1/CODE/checkpoints"
learning_rate_decrease_itr = 10
decay_factor = 0.94

display_freq = 50     #display_freq batches the training progress 
save_epoch_freq = 50
save_latest_freq = 5000
validation_freq = 100

# weights of loss
lambda_b = 10
lambda_s = 1
lambda_g = 0.01
lambda_p = 1


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

from tensorboardX import SummaryWriter
writer = SummaryWriter()

# call the frames data from the folders
resnet18 = models.resnet18(pretrained=True)
visual_net = VisualNet(resnet18)

audio_net = AudioNet()
audio_net.apply(weights_init)

# construct our models
model_backbone = BackboneModel(audio_net)


# use models with gpu
model_backbone = torch.nn.DataParallel(model_backbone, device_ids=gpu_ids)
model_backbone.to(dataset.device)

#define Adam optimzer
param_backbone = [{'params': visual_net.parameters(), 'lr': backbone_lr},
                {'params': audio_net.parameters(), 'lr': backbone_lr}]
optimizer_resnet = torch.optim.Adam(visual_net.parameters(), lr, param_backbone, betas=(beta1,0.999), weight_decay=weight_decay)
optimizer_backbone = torch.optim.Adam(param_backbone, betas=(beta1,0.999), weight_decay=weight_decay)

# set up loss function
loss_criterion = torch.nn.MSELoss()
if(len(gpu_ids) > 0):
    loss_criterion.cuda(gpu_ids[0])

batch_loss = []
total_steps = 0

for epoch in range(epochs):
    torch.cuda.synchronize()
    for i, data in enumerate(data_loader):
        
                total_steps += batch_size

                # forward pass
                visual_net.zero_grad()
                model_backbone.zero_grad()
                
                visual_input = data['frame']
                visual_feature = visual_net.forward(visual_input)
                
                output_backbone = model_backbone.forward(data, visual_feature)


                # compute loss for each model
                difference_loss = loss_criterion(output_backbone['binaural_spectrogram'], Variable(output_backbone['audio_gt'], requires_grad=False))
                channel1_loss = loss_criterion(output_backbone['channel1_pred'], data['channel1_spec'])
                channel2_loss = loss_criterion(output_backbone['channel2_pred'], data['channel2_spec'])
                loss_backbone = difference_loss + channel1_loss + channel2_loss
                
                
                # combine loss
                loss = lambda_b * loss_backbone
                batch_loss.append(loss.item())

                # update optimizer
                optimizer_resnet.zero_grad()
                optimizer_backbone.zero_grad()
                
                loss_backbone.backward()
                
                optimizer_resnet.step()
                optimizer_backbone.step()



                if(i % display_freq == 0):
                        print('Display training progress at (epoch %d, total steps %d)' % (epoch, total_steps))
                        avg_loss = sum(batch_loss) / len(batch_loss)
                        print('Average loss: %.3f' % (avg_loss))
                        batch_loss = []
                        writer.add_scalar('data/loss', avg_loss, total_steps)
                        

                if(i % save_latest_freq == 0):
                        print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                        torch.save(visual_net.state_dict(), os.path.join('.', checkpoints_dir, 'visual_latest.pth'))
                        torch.save(audio_net.state_dict(), os.path.join('.', checkpoints_dir, 'audio_latest.pth'))

                if(i % validation_freq == 0):
                        model_backbone.eval()
                        dataset.mode = 'val'
                        print('Display validation results at (epoch %d, total_steps %d)' % (epoch, total_steps))
                        val_err = display_val(model_backbone, loss_criterion, writer, total_steps, data_loader_val, opt)
                        print('end of display \n')
                        model_backbone.train()
                        dataset.mode = 'train'
                        #save the model that achieves the smallest validation error
                        if val_err < best_err:
                            best_err = val_err
                            print('saving the best model (epoch %d, total_steps %d) with validation error %.3f\n' % (epoch, total_steps, val_err))
                            torch.save(visual_net.state_dict(), os.path.join('.', checkpoints_dir, 'visual_best.pth'))
                            torch.save(audio_net.state_dict(), os.path.join('.', checkpoints_dir, 'audio_best.pth'))

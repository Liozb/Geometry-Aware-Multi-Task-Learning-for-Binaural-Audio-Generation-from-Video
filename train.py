# this code handles the train
from Models.backbone_model import *
from Models.geometry_model import *
from Models.spatial_model import *
from Models.rir_model import *
from Models.model import *
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
    
    
gpu_avilibale = True
if not gpu_avilibale:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

frames_dir = "/dsi/gannot-lab2/datasets2/FAIR-Play/frames_30fps/"
audios_dir = "/dsi/gannot-lab2/datasets2/FAIR-Play/binaural_audios/"
batch_size = 64
epochs = 1000
gpu_ids = [0,1,2,3]
lr = 1e-4
lr_big = 1e-3 
beta1 = 0.9
weight_decay = 0.0005 # use for regolization
train_epochs = 1000
checkpoints_dir = "/dsi/bermanl1/CODE/checkpoints"
learning_rate_decrease_itr = 10
decay_factor = 0.94
alpha = 0

display_freq = 50     #display_freq batches the training progress 
save_epoch_freq = 50
save_latest_freq = 5000
validation_freq = 100

# weights of loss
lambda_b = 10
lambda_s = 1
lambda_g = 0.01
lambda_p = 1


dataset = AudioVisualDataset(audios_dir, frames_dir, gpu_avilibale)
data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(dataset.nThreads))


# validation dataset
dataset.mode = 'val'
val_dataset = AudioVisualDataset(audios_dir, frames_dir, gpu_avilibale)
data_loader_val = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(val_dataset.nThreads))
dataset.mode = 'train'

from tensorboardX import SummaryWriter
writer = SummaryWriter()

## build nets
# resnet18 main net in our code
resnet18 = models.resnet18(pretrained=True)
visual_net = VisualNet(resnet18)

# spatial coherence net
spatial_net = AudioNet(input_nc=4)
spatial_net.apply(weights_init)

# audio network for backbone
audio_net = AudioNet()
audio_net.apply(weights_init)

# fusion network for backbone
fusion_net = APNet()
fusion_net.apply(weights_init)

# generator net for rir (Not used for FairPlay dataset)
generator = Generator()
generator.apply(weights_init)

nets = (visual_net, spatial_net, audio_net, fusion_net, generator)

# construct our models
model = model(nets)

# use models with gpu
if gpu_avilibale:
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.to(dataset.device)
    
else:
    model.to('cpu')
    
sum = sum([p.numel() for p in model.parameters() if p.requires_grad])

print("the number of parametrs is:",sum)
    
     
#define Adam optimzer
param_backbone = [{'params': visual_net.parameters(), 'lr': lr},
                {'params': audio_net.parameters(), 'lr': lr_big},
                {'params': fusion_net.parameters(), 'lr': lr_big},
                {'params': spatial_net.parameters(), 'lr': lr}]
#optimizer_resnet = torch.optim.Adam(visual_net.parameters(), lr, param_backbone, betas=(beta1,0.999), weight_decay=weight_decay)
optimizer = torch.optim.Adam(param_backbone, betas=(beta1,0.999), weight_decay=weight_decay)

# set up loss function
loss_criterion = torch.nn.MSELoss()
spatial_loss_criterion = torch.nn.BCELoss()
if(len(gpu_ids) > 0 and gpu_avilibale):
    loss_criterion.cuda(gpu_ids[0])

batch_loss = []
total_steps = 0

for epoch in range(epochs):
    if gpu_avilibale:
        torch.cuda.synchronize()
    for i, data in enumerate(data_loader):
        
                total_steps += batch_size

                ## forward pass
                # zero grad
                optimizer.zero_grad()
                
                output = model(data)
               

                ## compute loss for each model
                # backbone loss
                
                difference_loss = loss_criterion(output['binaural_spectrogram'], Variable(output['audio_gt'], requires_grad=False))
                channel1_loss = loss_criterion(output['left_spectrogram'], data['channel1_spec'][:,:,:-1,:])
                channel2_loss = loss_criterion(output['right_spectrogram'], data['channel2_spec'][:,:,:-1,:])
                loss_backbone = difference_loss + channel1_loss + channel2_loss
                
                # geometric consistency loss
                mse_geometry = loss_criterion(output['visual_feature'], output['second_visual_feature']) 
                loss_geometry = np.max(mse_geometry - alpha, 0)
                
                # spatial coherence loss
                c = output['c']
                c_pred = output['c_pred']
                loss_spatial = spatial_loss_criterion(c, c_pred)
                
                # combine loss
                loss = lambda_b * loss_backbone + lambda_g * loss_geometry + lambda_s * loss_spatial
                batch_loss.append(loss.item())

                # update optimizer
                #optimizer_resnet.zero_grad()
                optimizer.zero_grad()
                
                loss.backward()
                
                #optimizer_resnet.step()
                optimizer.step()



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
                            torch.save(visual_net.state_dict(), os.path.join('.', checkpoints_dir, 'visual_best.pth'))
                            torch.save(audio_net.state_dict(), os.path.join('.', checkpoints_dir, 'audio_best.pth'))


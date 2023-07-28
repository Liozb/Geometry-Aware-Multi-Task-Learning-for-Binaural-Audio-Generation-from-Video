# this code handles the train
from Models.backbone_model import *
from Models.geometry_model import *
from Models.spatial_model import *
from Models.rir_model import *
from Models.model import *
from imports import * 
from Datasets.AudioVisualDataset import *
from networks.Networks import *
from params import *

def data_test_handle(data, idx):
        frames = data['frames']
        audio_mix = data['channel1_spec']
        audio_channel1 = data['audio_channel1']
        audio_channel2 = data['audio_channel2']
        
        audio_start_time = idx * test_overlap * audio_length
        audio_end_time = idx * test_overlap * audio_length + audio_length
        
        # get the closest frame to the audio segment
        frame_index = int(round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10))  # 10 frames extracted per second
        frame = frames[frame_index]
  

        # get a frame 1 secend befor/after the original frame
        delta = random.uniform(-1, 1)
        second_frame_index = int(np.round(frame_index + 10*delta)) 
        if second_frame_index <= 0:
            second_frame_index = int(np.round(frame_index + 10*abs(delta)))
        second_frame = frames[second_frame_index]
        
        # passing the spectrogram of the difference
        audio_diff_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 - audio_channel2))
        audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio_mix))
        channel1_spec = torch.FloatTensor(generate_spectrogram(audio_channel1))
        channel2_spec = torch.FloatTensor(generate_spectrogram(audio_channel2))
        
        return {'frame': frame, 'second_frame': second_frame, 'audio_diff_spec': audio_diff_spec, 'audio_mix_spec': audio_mix_spec, 'channel1_spec': channel1_spec , 'channel2_spec': channel2_spec}


def debug_dataset(dataset, epoch, idx=15,flag='input'):
    """ debug function to whatch a specific index in the dataset.
        saves the output to the dubug folder. 

    Args:
        idx (int, optional): index for a place in the dataset. Defaults to 15.
    """
    if flag == 'input':
        frame = dataset['frame']
        frame_idx = frame[idx]
        
        data_idx = dataset['audio_mix_spec']
        audio_spec_idx = data_idx[idx]
    elif flag == 'output':
        data_idx = dataset['binaural_spectrogram']
        audio_spec_idx = data_idx[idx]
        
        cpu_tensor = audio_spec_idx.clone().cpu()
        audio_spec_idx = cpu_tensor.detach()

    audio_spec_idx = torch.sqrt(audio_spec_idx[0,:,:]**2 + audio_spec_idx[1,:,:]**2) 
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(audio_spec_idx), ref=np.max), hop_length=160,
                            y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB)')
    if flag == 'input':
        plt.savefig('pic_for_debug/audio_spec_input_' + str(epoch) + '.jpg', format='jpg')
        
        plt.imshow(frame_idx.permute(1,2,0).numpy())
        plt.savefig('pic_for_debug/frame.jpg', format='jpg')
    elif flag == 'output':
        plt.savefig('pic_for_debug/audio_spec_output_' + str(epoch) + '.jpg', format='jpg')


    
def display_val(model, loss_criterion, writer, index, dataset_val):
    losses = []
    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            output = model(val_data, mode='val')
            channel1_spec = val_data['channel1_spec'].to(device)
            channel2_spec = val_data['channel2_spec'].to(device)
            fusion_loss1 = loss_criterion(output['left_spectrogram'], channel1_spec[:,:,:-1,:])
            fusion_loss2 = loss_criterion(output['right_spectrogram'], channel2_spec[:,:,:-1,:])
            fus_loss = (fusion_loss1 / 2 + fusion_loss2 / 2)
            loss = loss_criterion(output['binaural_spectrogram'], Variable(output['audio_gt'])) + fus_loss

            losses.append(loss.item()) 
    avg_loss = sum(losses)/len(losses)
    writer.add_scalar('data/val_loss', avg_loss, index)
    print('val loss: %.3f' % avg_loss)
    return avg_loss


def clear_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Iterate over the files and subdirectories in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file or subdirectory
        if os.path.isfile(file_path):
            # If it's a file, remove it
            os.remove(file_path)
            print(f"Removed file: {file_path}")
        elif os.path.isdir(file_path):
            # If it's a subdirectory, recursively clear its contents and then remove it
            clear_folder(file_path)
            os.rmdir(file_path)
            print(f"Removed directory: {file_path}")
    

if __name__ == '__main__':
    
    clear_folder(debug_dir)
    
    dataset = AudioVisualDataset(audios_dir, frames_dir, gpu_available)
    subset_dataset = Subset(dataset, dataset.train_indices)
    data_loader = DataLoader(
                subset_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=int(dataset.nThreads))


        # validation dataset
    dataset.mode = 'val'
    val_dataset = AudioVisualDataset(audios_dir, frames_dir, gpu_available,  'val')
    subset_val_dataset = Subset(val_dataset, val_dataset.val_indices)
    data_loader_val = DataLoader(
                subset_val_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=int(val_dataset.nThreads))
    dataset.mode = 'train'
    
    # test dataset
    test_dataset = AudioVisualDataset(audios_dir, frames_dir, gpu_available, 'test')
    subset_test_dataset = Subset(test_dataset, test_dataset.test_indices)
    data_loader_test = DataLoader(
                subset_test_dataset,
                batch_size=batch_size_test,
                shuffle=True,
                num_workers=int(test_dataset.nThreads))
    

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
    if gpu_available:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        model.to(dataset.device)
    else:
        model.to('cpu')
        
    param_sum = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("the number of parametrs is:",param_sum)
        
        
    #define Adam optimzer
    param_backbone = [{'params': visual_net.parameters(), 'lr': lr},
                    {'params': audio_net.parameters(), 'lr': lr_big},
                    {'params': fusion_net.parameters(), 'lr': lr_big},
                    {'params': spatial_net.parameters(), 'lr': lr}]
    
    optimizer = torch.optim.Adam(param_backbone, betas=(beta1,0.999), weight_decay=weight_decay)

    # set up loss function
    loss_criterion = torch.nn.MSELoss()
    spatial_loss_criterion = torch.nn.BCEWithLogitsLoss()
    if(len(gpu_ids) > 0 and gpu_available):
        loss_criterion.cuda(gpu_ids[0])

    batch_loss, batch_loss1, batch_fusion_loss, batch_rir_loss, batch_spat_const_loss, batch_geom_const_loss = [], [], [], [], [], []
    total_steps = 0
    best_err = float("inf")

    for epoch in range(epochs):
        if gpu_available:
            torch.cuda.synchronize(device=device)
        for i, data in enumerate(data_loader):

                total_steps += batch_size

                ## forward pass
                # zero grad
                optimizer.zero_grad()

                output = model(data,mode='train')
                

                ## compute loss for each model
                # backbone loss
                channel1_spec = data['channel1_spec'].to(device)
                channel2_spec = data['channel2_spec'].to(device)
                difference_loss = loss_criterion(output['binaural_spectrogram'], output['audio_gt'])
                channel1_loss = loss_criterion(output['left_spectrogram'], channel1_spec[:,:,:-1,:])
                channel2_loss = loss_criterion(output['right_spectrogram'], channel2_spec[:,:,:-1,:])
                fusion_loss = (channel1_loss / 2 + channel2_loss / 2)
                loss_backbone = lambda_binarual * difference_loss + lambda_f * fusion_loss
                
                # geometric consistency loss
                mse_geometry = loss_criterion(output['visual_feature'], output['second_visual_feature']) 
                loss_geometry = torch.maximum(mse_geometry - alpha, torch.tensor(0))
                
                # spatial coherence loss
                c = output['cl_pred']
                c_pred = output['label']
                loss_spatial = spatial_loss_criterion(c, c_pred)
                
                # combine loss
                loss = lambda_b * loss_backbone + lambda_g * loss_geometry + lambda_s * loss_spatial
                batch_loss.append(loss.item())
                batch_loss1.append(difference_loss.item())
                batch_fusion_loss.append(fusion_loss.item())
                batch_spat_const_loss.append(loss_spatial.item())
                batch_geom_const_loss.append(loss_geometry.item())
                
                # update optimizer
                #optimizer_resnet.zero_grad()
                optimizer.zero_grad()
                
                loss.backward()
                
                #optimizer_resnet.step()
                optimizer.step()



                if(i % display_freq == 0):
                    debug_dataset(data, epoch)
                    debug_dataset(output, epoch, flag='output')
                    
                    print('Display training progress at (epoch %d, total_steps %d)' % (epoch, total_steps))
                    avg_loss = sum(batch_loss) / len(batch_loss)
                    avg_loss1 = sum(batch_loss1) / len(batch_loss1)
                    avg_fusion_loss = sum(batch_fusion_loss) / len(batch_fusion_loss)
                    avg_spat_const_loss = sum(batch_spat_const_loss) / len(batch_spat_const_loss)
                    avg_geom_const_loss = sum(batch_geom_const_loss) / len(batch_geom_const_loss)
                    print('Average loss: %.3f' % (avg_loss))
                    batch_loss, batch_loss1, batch_fusion_loss, batch_rir_loss, batch_spat_const_loss, batch_geom_const_loss = [], [], [], [], [], []
                    writer.add_scalar('data/loss', avg_loss, total_steps)
                    writer.add_scalar('data/loss1', avg_loss1, total_steps)
                    writer.add_scalar('data/fusion_loss', avg_fusion_loss, total_steps)
                    writer.add_scalar('data/spat_const_loss', avg_spat_const_loss, total_steps)
                    writer.add_scalar('data/geom_const_loss', avg_geom_const_loss, total_steps)
                        

                if(i % save_latest_freq == 0):
                        print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                        torch.save(visual_net.state_dict(), os.path.join('.', checkpoints_dir, 'visual_latest.pth'))
                        torch.save(audio_net.state_dict(), os.path.join('.', checkpoints_dir, 'audio_latest.pth'))
                        torch.save(fusion_net.state_dict(), os.path.join('.', checkpoints_dir, 'fusion_latest.pth'))
                        torch.save(spatial_net.state_dict(), os.path.join('.', checkpoints_dir, 'classifier_latest.pth'))
                        torch.save(generator.state_dict(), os.path.join('.', checkpoints_dir, 'generator_latest.pth'))

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
                            torch.save(fusion_net.state_dict(), os.path.join('.', checkpoints_dir, 'fusion_best.pth'))
                            torch.save(spatial_net.state_dict(), os.path.join('.', checkpoints_dir, 'classifier_best.pth'))
                            torch.save(generator.state_dict(), os.path.join('.', checkpoints_dir, 'generator_best.pth'))
    # run the test
    model.eval()
    losses = []
    for idx, data in enumerate(data_loader_test):
        sliding_window_size = audio_length  * audio_sampling_rate
        audio_length_test = data['audio_mix'].shape[1]
        
        num_loops =  (np.floor(audio_length_test/sliding_window_size))/test_overlap - 1 + round(audio_length_test % sliding_window_size)
        
        for j in range(num_loops):
            
            test_data = data_test_handle(data, j)
        
            # Perform forward pass
            output = model(data, mode='test')
            
            
            channel1_spec = data['channel1_spec'].to(device)
            channel2_spec = data['channel2_spec'].to(device)
            difference_loss = loss_criterion(output['binaural_spectrogram'], output['audio_gt'])
            channel1_loss = loss_criterion(output['left_spectrogram'], channel1_spec[:,:,:-1,:])
            channel2_loss = loss_criterion(output['right_spectrogram'], channel2_spec[:,:,:-1,:])
            fusion_loss = (channel1_loss / 2 + channel2_loss / 2)
            loss_backbone = lambda_binarual * difference_loss + lambda_f * fusion_loss
            
            # geometric consistency loss
            mse_geometry = loss_criterion(output['visual_feature'], output['second_visual_feature']) 
            loss_geometry = np.max(mse_geometry - alpha, 0)
            
            # combine loss
            loss = lambda_b * loss_backbone + lambda_g * loss_geometry 
            losses.append(loss.item())
    loss_avg = (sum(losses)/len(losses)) 
    print("test average loss is:", loss_avg)       
    
            


    
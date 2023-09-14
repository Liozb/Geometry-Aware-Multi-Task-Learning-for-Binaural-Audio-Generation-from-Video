from Models.backbone_model import *
from Models.geometry_model import *
from Models.spatial_model import *
from Models.rir_model import *
from Models.model import *
from imports import * 
from Datasets.AudioVisualDataset import *
from networks.Networks import *
from params import *


def show_spec(spec_left_true,spec_right_true, spec_left_pred,spec_right_pred, idx):
    spec_left_pred = torch.clone(spec_left_pred).to('cpu').detach()
    spec_right_pred = torch.clone(spec_right_pred).to('cpu').detach()
    spec_left_true = torch.sqrt(spec_left_true[0,:,:]**2 + spec_left_true[1,:,:]**2) 
    spec_right_true = torch.sqrt(spec_right_true[0,:,:]**2 + spec_right_true[1,:,:]**2) 
    spec_left_pred = torch.sqrt(spec_left_pred[0,:,:]**2 + spec_left_pred[1,:,:]**2) 
    spec_right_pred = torch.sqrt(spec_right_pred[0,:,:]**2 + spec_right_pred[1,:,:]**2) 
    
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(spec_left_true), ref=np.max), hop_length=160,
                             y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Left True Spectrogram (dB)')
    
    plt.subplot(2, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(spec_right_true), ref=np.max), hop_length=160,
                             y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Right True Spectrogram (dB)')
    
    plt.subplot(2, 2, 3)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(spec_left_pred), ref=np.max), hop_length=160,
                             y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Left Predicted Spectrogram (dB)')
    
    plt.subplot(2, 2, 4)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(spec_right_pred), ref=np.max), hop_length=160,
                             y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Right Predicted Spectrogram (dB)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(test_spec_path,'spectrogram_subplot_' + str(idx) + '.jpg'), format='jpg')


def save_audio(audio_data, path):
    if len(audio_data) == 1:
        audio_data = audio_data/1.1/torch.max(torch.abs(audio_data))
        wavfile.write(path, audio_sampling_rate , audio_data.numpy().astype(np.float32))
    elif len(audio_data) == 2:
        audio_data[0] = audio_data[0]/1.1/torch.max(torch.abs(audio_data[0]))
        audio_data[1] = audio_data[1]/1.1/torch.max(torch.abs(audio_data[1]))
        if type(audio_data[0]).__module__ == np.__name__ and type(audio_data[1]).__module__ == np.__name__:
            audio_channel_left = audio_data[0]
            audio_channel_right = audio_data[1]
        if type(audio_data[0]).__module__ != np.__name__:
            audio_channel_left = audio_data[0].numpy().astype(np.float32)
        if type(audio_data[1]).__module__ != np.__name__:
            audio_channel_right = audio_data[1].numpy().astype(np.float32)
        
        audio =  np.hstack((audio_channel_left.reshape(-1, 1), audio_channel_right.reshape(-1, 1)))
        wavfile.write(path,audio_sampling_rate, audio)

    
def video_from_frames(frames, path):
    frame_duration = 1 / 30
    desired_duration = 10
    
    # Calculate the number of frames needed to achieve the desired duration
    num_frames = int(desired_duration / frame_duration)

    # Load the frames and create a list of video clips
    transposed_frames = [np.transpose(frame[0, :, :, :], (1, 2, 0)) for frame in frames]
    frame_clips = [mpy.ImageClip(frame_file, duration=frame_duration) for frame_file in transposed_frames[:num_frames]]

    # Concatenate the video clips to create the final video
    video_clip = mpy.concatenate_videoclips(frame_clips, method="compose")

    # Set the duration of the video
    video_duration = len(frame_clips) * frame_duration
    video_clip = video_clip.set_duration(video_duration)

    # Set the frame rate of the video
    frame_rate = 30
    video_clip = video_clip.set_fps(frame_rate)

    # Write the final video to the output file
    video_clip.write_videofile(path, codec='libx264')

    
    
def video_from_files(idx, paths):
    
    output_video = 'video_' + str(idx) + '.mp4'
    output_path = os.path.join(video_path, output_video)
    
    if os.path.exists(output_path):
            os.remove(output_path)
            
    combine_audio_path = paths["combine_audio"]

    # loading video gfg
    video_clip = mpy.VideoFileClip(paths["frames_video"]) 
    
    # create stereo audio
    audio_clip = mpy.AudioFileClip(combine_audio_path)

    video_clip = video_clip.set_audio(audio_clip)
    
    video_clip.write_videofile(output_path, codec='libx264')

    
    
def remove_temps_media(idx, paths):
    paths = list(paths.values())
    
    for file_path in paths:
            # Check if the file exists before removing it
        if os.path.exists(file_path):
            os.remove(file_path)


def create_video(frames, audio_left, audio_right, idx):
    frames_video_file = 'frame_video_' + str(idx) + '.mp4'
    combine_audio_file = 'combine_audio' + str(idx) + '.wav'
    
    frames_video_path = os.path.join(video_path, frames_video_file)
    combine_audio = os.path.join(video_path, combine_audio_file)
    
    paths = {"frames_video":frames_video_path, "combine_audio":combine_audio}
    
    # Create temp audio files and video separately
    save_audio([audio_left, audio_right], paths["combine_audio"])
    video_from_frames(frames, paths["frames_video"])
    
    # Combine audio files and video
    video_from_files(idx, paths)
    
    # Remove temp files
    remove_temps_media(idx, paths)  
    


def inverse_spectrogram(audio_spec, length,  type='tensor'):
    if type == 'tensor':
        audio_spectogram = audio_spec[0,0,:,:] + 1j * audio_spec[0,1,:,:]
    else:
        audio_spectogram = audio_spec[0] + 1j * audio_spec[1]
        
    audio_spectogram = audio_spectogram.detach().cpu().numpy()
    
    # Compute the inverse STFT to get the audio signal
    audio_reconstructed = librosa.core.istft(audio_spectogram, hop_length=160, win_length=400, length=length, center=True)
    
    return audio_reconstructed


def data_test_handle(data, idx, num_loops):
        frames = data['frames']
        audio_mix = data['audio_mix']
        audio_channel1 = data['audio_channel1']
        audio_channel2 = data['audio_channel2']
        audio_full_time = np.floor(len(audio_channel1[0,:]) / audio_sampling_rate)
        
        if idx < num_loops - 1:
            audio_start_time = idx * test_overlap * audio_length
            audio_end_time = idx * test_overlap * audio_length + audio_length
            audio_start = int(round(audio_start_time * audio_sampling_rate))
            audio_end = int(round(audio_end_time * audio_sampling_rate))
        else:
            audio_start_time = audio_full_time - audio_length
            audio_end_time = audio_full_time
            audio_start = int(round(audio_start_time * audio_sampling_rate))
            audio_end = int(round(audio_end_time * audio_sampling_rate))
            
        
        # get the closest frame to the audio segment
        frame_index = int(round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10))  # 10 frames extracted per second
        frame = frames[frame_index]
  

        # get a frame 1 secend befor/after the original frame
        delta = random.uniform(-1, 1)
        second_frame_index = int(np.round(frame_index + 10*delta)) 
        if second_frame_index <= 0:
            second_frame_index = int(np.round(frame_index + 10*abs(delta)))
        second_frame = frames[second_frame_index]
        
        
        audio_mix = audio_mix[:, audio_start : audio_end]
        audio_channel1 = audio_channel1[:, audio_start : audio_end]
        audio_channel2 = audio_channel2[:, audio_start : audio_end]
        
        # passing the spectrogram of the difference
        audio_diff_spec = batch_spec(audio_channel1 - audio_channel2)
        audio_mix_spec = batch_spec(audio_mix)
        channel1_spec = batch_spec(audio_channel1)
        channel2_spec = batch_spec(audio_channel2)
        
        return {'frame': frame, 'second_frame': second_frame, 'audio_diff_spec': audio_diff_spec, 'audio_mix_spec': audio_mix_spec, 'channel1_spec': channel1_spec , 'channel2_spec': channel2_spec}


def handle_output(data, outputs, idx, num_loops):
    left_spectrogram_slide = data['left_spectrogram']
    right_spectrogram_slide = data['right_spectrogram']
    binaural_spectrogram_slide = data['binaural_spectrogram']
    audio_gt_slide = data['audio_gt']
    time_frame = audio_gt_slide.shape[3]
    
    
    left_spectrogram = outputs["left_spectrogram"]
    right_spectrogram = outputs["right_spectrogram"]
    binaural_spectrogram = outputs["binaural_spectrogram"]
    audio_gt = outputs["audio_gt"]
    
    if idx == num_loops - 1:
        left_spectrogram[:,:,:,int(left_spectrogram.shape[3] - time_frame) : ] += left_spectrogram_slide
        left_spectrogram[:,:,:, int(time_frame * test_overlap) :int((idx - 1) * time_frame * test_overlap + time_frame) ] *= 0.5
        
        right_spectrogram[:,:,:,int(right_spectrogram.shape[3] - time_frame) : ] += right_spectrogram_slide
        right_spectrogram[:,:,:, int(time_frame * test_overlap) :int((idx - 1) * time_frame * test_overlap + time_frame) ] *= 0.5
        
        binaural_spectrogram[:,:,:,int(binaural_spectrogram.shape[3] - time_frame) : ] += binaural_spectrogram_slide
        binaural_spectrogram[:,:,:, int(time_frame * test_overlap) :int((idx - 1) * time_frame * test_overlap + time_frame) ] *= 0.5
        
        audio_gt[:,:,:,int(audio_gt.shape[3] - time_frame) : ] += audio_gt_slide
        audio_gt[:,:,:, int(time_frame * test_overlap) :int((idx - 1) * time_frame * test_overlap + time_frame) ] *= 0.5
        
        if left_spectrogram.shape[3] - time_frame < (idx - 2) * time_frame * test_overlap + time_frame:
            left_spectrogram[:,:,:, int(left_spectrogram.shape[3] - time_frame) :int((idx - 2) * time_frame * test_overlap + time_frame) ] *= 2/3
            right_spectrogram[:,:,:, int(right_spectrogram.shape[3] - time_frame) :int((idx - 2) * time_frame * test_overlap + time_frame) ] *= 2/3
            binaural_spectrogram[:,:,:, int(binaural_spectrogram.shape[3] - time_frame) :int((idx - 2) * time_frame * test_overlap + time_frame) ] *= 2/3
            audio_gt[:,:,:, int(audio_gt.shape[3] - time_frame) :int((idx - 2) * time_frame * test_overlap + time_frame) ] *= 2/3

        
    else:
        left_spectrogram[:,:,:,int(idx * time_frame * test_overlap) :int(idx * time_frame * test_overlap + time_frame) ] += left_spectrogram_slide
        right_spectrogram[:,:,:,int(idx * time_frame * test_overlap) :int(idx * time_frame * test_overlap + time_frame) ] += right_spectrogram_slide
        binaural_spectrogram[:,:,:,int(idx * time_frame * test_overlap) :int(idx * time_frame * test_overlap + time_frame) ] += binaural_spectrogram_slide
        audio_gt[:,:,:,int(idx * time_frame * test_overlap) :int(idx * time_frame * test_overlap + time_frame) ] += audio_gt_slide
    
    return (left_spectrogram, right_spectrogram, binaural_spectrogram, audio_gt)


def build_test_model():
    resnet18 = models.resnet18(pretrained=True)
    visual_net = VisualNet(resnet18).to(device)
    audio_net = AudioNet().to(device)
    fusion_net = APNet().to(device)
    spatial_net = AudioNet(input_nc=4).to(device)
    generator = Generator().to(device)

    # Load the saved model parameters
    visual_net.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'visual_best.pth')))
    audio_net.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'audio_best.pth')))
    fusion_net.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'fusion_best.pth')))
    spatial_net.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'classifier_best.pth')))
    generator.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'generator_best.pth')))

    nets = (visual_net, spatial_net, audio_net, fusion_net, generator)

    # construct our models
    test_model = model(nets)
    
    return test_model


def batch_spec(batch_audio):
    audio = batch_audio[0,:].numpy()
    spec = torch.FloatTensor(generate_spectrogram(audio))
    batch_spec = torch.unsqueeze(spec, dim=0)
    return batch_spec


if __name__=='__main__':
    
    
    test_dataset = AudioVisualDataset(audios_dir, frames_dir, gpu_available, 'test')
    subset_test_dataset = Subset(test_dataset, test_dataset.test_indices)
    data_loader_test = DataLoader(
                subset_test_dataset,
                batch_size=batch_size_test,
                shuffle=True,
                num_workers=int(test_dataset.nThreads))
    
    
    # run the test
    test_model = build_test_model()
    test_model.eval()
    loss_criterion = torch.nn.MSELoss()
    if(len(gpu_ids) > 0 and gpu_available):
        loss_criterion.cuda(gpu_ids[0])
    losses_stft, losses_env = [],[]
    batch_idx = 0
    
    for idx, data in enumerate(data_loader_test):
        sliding_window_size = audio_length  * audio_sampling_rate
        audio_length_test = data['audio_mix'].shape[1]
        
        num_loops =  int((np.floor(audio_length_test/sliding_window_size))/test_overlap - 1 + np.ceil(audio_length_test / sliding_window_size - audio_length_test // sliding_window_size))
        loss_geometry = []
        
        channel1_spec = batch_spec(data['audio_channel1']).to(device)
        channel2_spec =batch_spec(data['audio_channel2']).to(device)
        
        left_spectrogram = torch.zeros_like(channel1_spec[:,:,:-1,:]).to(device)
        right_spectrogram = torch.zeros_like(channel1_spec[:,:,:-1,:]).to(device)
        binaural_spectrogram = torch.zeros_like(channel1_spec[:,:,:-1,:]).to(device)
        audio_gt = torch.zeros_like(channel1_spec[:,:,:-1,:]).to(device)
        outputs = {"left_spectrogram" : left_spectrogram, "right_spectrogram" : right_spectrogram, "binaural_spectrogram" : binaural_spectrogram, "audio_gt" : audio_gt}
        frames = []
        for j in range(num_loops):
            
            test_data = data_test_handle(data, j, num_loops)
        
            
            # Perform forward pass
            output = test_model(test_data, mode='test')
            
            (left_spectrogram, right_spectrogram, binaural_spectrogram, audio_gt) = handle_output(output, outputs, j, num_loops)

            # Show spectogram for debug
            if j == debug_test_idx and idx < 10:
                show_spec(test_data["channel1_spec"][0,:,:,:],test_data["channel2_spec"][0,:,:,:], output["left_spectrogram"][0,:,:,:],output["right_spectrogram"][0,:,:,:], idx)

            # geometric consistency loss
            mse_geometry = loss_criterion(output['visual_feature'], output['second_visual_feature']) 
            loss_geometry_slide = torch.maximum(mse_geometry - alpha, torch.tensor(0))
            
            loss_geometry.append(loss_geometry_slide.item())
        
        # Reconstruct to time domain
        re_left_spectrogram = inverse_spectrogram(left_spectrogram, audio_length_test)
        re_left_spectrogram = torch.FloatTensor(re_left_spectrogram)
        re_right_spectrogram = inverse_spectrogram(left_spectrogram, audio_length_test)
        re_right_spectrogram = torch.FloatTensor(re_right_spectrogram)
        
        # Compute loss STFT
        loss_geometry = sum(loss_geometry) / len(loss_geometry)
        difference_loss = loss_criterion(binaural_spectrogram, audio_gt)
        channel1_loss = loss_criterion(left_spectrogram , channel1_spec[:,:,:-1,:])
        channel2_loss = loss_criterion(right_spectrogram, channel2_spec[:,:,:-1,:])
        fusion_loss = 0.5 * (channel1_loss + channel2_loss)
        loss_backbone = lambda_binarual * difference_loss + lambda_f * fusion_loss
        
        
        # combine loss STFT
        loss = lambda_b * loss_backbone + lambda_g * loss_geometry 
        losses_stft.append(loss_backbone.item())
        
        # Cumpute loss ENV
        loss_left = loss_criterion(re_left_spectrogram, data['audio_channel1'][0,:])
        loss_right = loss_criterion(re_right_spectrogram, data['audio_channel2'][0,:])
        
        env_loss = 0.5 * (loss_left + loss_right)
        losses_env.append(env_loss.item())
        
        # Create output video
        frames = data["frames_to_video"]
        if idx < 10:
            create_video(frames, re_left_spectrogram, re_right_spectrogram, batch_idx)
        
        batch_idx += 1
        
    loss_avg_stft = (sum(losses_stft)/len(losses_stft)) 
    print("test average loss (stft) is:", loss_avg_stft)   
        
    loss_avg_env = (sum(losses_env)/len(losses_env)) 
    print("test average loss(env) is:", loss_avg_env)   
    
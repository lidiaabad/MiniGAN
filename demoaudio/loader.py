import math
import random

import torch
import numpy as np
import pandas as pd

import utils.audioutils as au
from se import augment


def load_train_partitions(path, window_size=24000, fs=16000, snr_range=[5, 30], augments=None):
    
    # Load train set
    train_df = pd.read_csv(path + 'train.tsv', sep='\t')
    train_noise_paths = train_df[train_df['Sound_Type'] != 'speech']['Sample_Path'].tolist() #store the paths to noise data of train set
    train_df = train_df[train_df['Sound_Type'] == 'speech'] #modify train_df so that it only contains the speech (that is the valentini part, no valentini_noises)
    train_ID_list = list(train_df['Sample_ID'])

    # Load validation set
    validation_df = pd.read_csv(path + 'dev.tsv', sep='\t')
    validation_noise_paths = validation_df[validation_df['Sound_Type'] != 'speech']['Sample_Path'].tolist()
    validation_df = validation_df[validation_df['Sound_Type'] == 'speech'] #the same for the dev test 
    validation_ID_list = list(validation_df['Sample_ID'])

    # Generate Datasets 
    # with the ID list of the noisy datasets (no only noises), the df, the window size, the freq, the noise_paths, the snr range  
    train_dataset = AudioDataset(
        train_ID_list,
        train_df,
        window_size=window_size, #size of the audio window used during data preprocessing
        fs=fs,
        noise_paths=train_noise_paths, #the sampling frequency of the audio data.
        snr_range=snr_range,
        original_noise_prob=0.0, #bability of using original noise samples (from train_noise_paths) during data augmentation. A value of 0.0 indicates that only synthesized or augmented noise will be used.
        augments=augments
        )
    validation_dataset = AudioDataset(
        validation_ID_list,
        validation_df,
        window_size=window_size,
        fs=fs,
        noise_paths=validation_noise_paths,
        snr_range=snr_range,
        original_noise_prob=0.0,#WHY 0 here and in 
        augments=augments
        )

    return train_dataset, validation_dataset

def load_test_partition(path, window_size=24000, fs=16000, snr_range=[5, 30], augments=None):
    
    # Load test set
    test_df = pd.read_csv(path + 'test.tsv', sep='\t')
    test_noise_paths = test_df[test_df['Sound_Type'] != 'speech']['Sample_Path'].tolist() #none
    test_df = test_df[test_df['Sound_Type'] == 'speech'] #the same but here it will have no effect as there is no diferent to speech. It should tough
    test_ID_list = list(test_df['Sample_ID'])

    # Generate Dataset
    test_dataset = AudioDataset(
        test_ID_list,
        test_df,
        window_size=window_size,
        fs=fs,
        noise_paths=test_noise_paths,
        snr_range=snr_range,
        original_noise_prob=1.0,
        augments=augments
        )

    return test_dataset


class AudioDataset(torch.utils.data.Dataset):
    """
    Torch dataset for lazy load.
    """
    def __init__(self, list_IDs, dataframe, window_size=24000, fs=16000, noise_paths=None,
        snr_range=[5, 30], original_noise_prob=1.0, augments=None):
        
        self.window_size = window_size
        self.fs = fs # Hz

        # Data information
        self.list_IDs = list_IDs
        self.dataframe = dataframe
        self.n_samples = len(list_IDs)

        # Add noise sound: can be an empty list
        self.snr_range = snr_range
        self.noise_paths = noise_paths

        # Data augments
        self.augments = [k for k,v in augments.items() if v == True]
        #self.white_noise = True if augments['white_noise'] else None
        self.bandmask = augment.BandMask() if augments['bandmask'] else None #in augment.py
        self.revecho = augment.RevEcho() if augments['revecho'] else None #in augment.py
        self.white_noise= augment.WhiteNoise() if augments['white_noise'] else None

        # Valentini usage of original noise prob
        self.original_noise_prob = original_noise_prob

    def __len__(self):
        """
        Denote dataset sample.
        """
        return len(self.list_IDs)

    def __repr__(self):
        """
        Data infromation
        """
        repr_str = (
            "Number of samples: " + str(self.n_samples) + "\n"
            "Window size: " + str(self.window_size) + "\n"
            "Augments: " + str(self.augments) + "\n"
            "SNR range: " + str(self.snr_range) + "\n"
            "Databases: " + str(np.unique(self.dataframe['Database'])) + "\n"

        )
        return repr_str
        
    def __getitem__(self, index):
        """
        Get a single sample

        Args:
            index: index to recover a single sample
        Returns:
            x,y: features extracted and label
        """
        # Select sample
        ID = self.list_IDs[index]

        # Read audio
        audio_path = self.dataframe.set_index('Sample_ID').at[ID, 'Sample_Path'] #extracts the path to apply read method on it
        clean = self.__read_wav(audio_path) #below

        # Prepare audio
        clean, noisy = self.__prepare_audio(clean, ID, audio_path) #below

        # Renormalize audio
        noisy = self.__normalize_audio(noisy, eps=0)
        clean= self.__normalize_audio(clean, eps=0)

        return ID, noisy, clean

    def __read_wav(self, filepath):
        """
        Read audio wave file applying normalization with respecto of the maximum of the signal

        Args:
            filepath: audio file path
        Returns:
            audio_signal: numpy array containing audio signal
        """
        _, audio_signal = au.open_wavfile(filepath) #using the method open_wavfile on the filepath (audio_path)from audioutils we get the audiosignal 
        audio_signal = self.__normalize_audio(audio_signal) #this audio is normalized to obtain the audio_signal
        return audio_signal
    
    def __normalize_audio(self, audio, eps=0.001):
        """
        Peak normalization.
        """
        return (audio.astype(np.float32) / float(np.amax(np.abs(audio)))) + eps 

    def __prepare_audio(self, audio_signal, ID, audio_path):
        """
        Apply all data transformations and get the noisy signal

        Args:
            audio_signal: audio sample to crop
            ID: of the sample
        """
        database = self.dataframe.set_index('Sample_ID').at[ID, 'Database'].lower() #converts the retrieved database name to lowercase. This is likely done for consistency, as it ensures that database names are treated uniformly regardless of capitalization.

        # Use noisy Valentini with a given probability
        valentini = False
        if('valentini' in database and np.random.uniform(0, 1) < self.original_noise_prob): #if prob 0 this does never happen
            valentini = True
            noisy_path = audio_path.replace('clean', 'noisy') #get noisty path by replacing word clean with noiy
            noisy_signal = self.__read_wav(noisy_path) #obtain the file
            if(noisy_signal.shape != audio_signal.shape):
                print('Problem with file: ' + noisy_path)

        # Adapt sample to windows size
        audio_length = audio_signal.shape[0] #calculates length of audi9
        if(audio_length >= self.window_size): #if it is bigger
            
            # If audio is bigger than window size use random crop: random shift
            left_bound = random.randint(0, audio_length - self.window_size) #calculates starting point of window
            right_bound = left_bound + self.window_size #eding point of window
            audio_signal = audio_signal[left_bound:right_bound] #isolate from left to right bound. 
            
            # Use the same random shift fo noisy data in valentini dataset 
            if (valentini):
                noisy_signal = noisy_signal[left_bound:right_bound]
        else:
            # If the audio is smaller than the window size: pad original signal with 0z
            padding = self.window_size - audio_length #calculates padding needed
            bounds_sizes = np.random.multinomial(padding, np.ones(2)/2, size=1)[0] #randomly distributes the padding equally on both sides
            #of the audio signal. The np.ones(2)/2 generates prob 0.5 for the begginign and 0.5 for the ending. size=1, the result is
            #an array with a single element containing the generated sample.
            audio_signal = np.pad( #apply padding
                audio_signal,
                (bounds_sizes[0], bounds_sizes[1]),
                'constant',
                constant_values=(0, 0)
                )
            if (valentini):
                noisy_signal = np.pad(
                    noisy_signal,
                    (bounds_sizes[0], bounds_sizes[1]),
                    'constant',
                    constant_values=(0, 0)
                    )

        # Add white noise
        if(self.white_noise):
            audio_signal = self.__normalize_audio(audio_signal)
            noisy_signal = self.white_noise(audio_signal)
        else: 
            noisy_signal = self.__normalize_audio(audio_signal)

        # Combine with background sound: 
        if(not valentini):
            noisy_signal = self.__add_background_noise(noisy_signal) #if valentini is not present we add bck noises

        # Bandmask: to Valentini data
        if(self.bandmask and 'valentini' in database):
            noisy_signal = self.bandmask(noisy_signal) #apply bandmask

        return audio_signal, noisy_signal


    def __add_background_noise(self, data):
        """
        Add background noise to a single sample.

        Args:
            data: original audio signal
        """
        background_noise_path = random.sample(self.noise_paths, 1)[0] #selects one elmenet from the paths of the noises of valentini. 
        background_noise = self.__read_wav(background_noise_path) #reads it 

        # Store len
        background_length = len(background_noise)
        audio_length = len(data)

        # Pad with 0z if background noise is shorter than audio
        #pad as before
        if background_length < audio_length:
            padding = audio_length - background_length
            bounds_sizes = np.random.multinomial(padding, np.ones(2)/2, size=1)[0]
            background_noise = np.pad(background_noise, (bounds_sizes[0], bounds_sizes[1]), 'constant', constant_values=(0, 0))
            background_length = len(background_noise)

        left_bound = random.randint(0, background_length - audio_length)
        right_bound = left_bound + audio_length

        background_noise = background_noise[left_bound:right_bound]
        background_noise=background_noise.astype(np.float64)
        
        # Calculate SNR
        SNR_DB = np.random.uniform(low=self.snr_range[0], high=self.snr_range[1]) #generate a random uniform SNR value
        data_rms = np.sqrt(np.mean(data**2))  #calculate root square of data audio (data is the oiriginal audio signal)
        background_noise_rms = np.sqrt(np.mean(background_noise**2)) #the same as before again to obtain the pw
        desired_noise_rms = np.sqrt((data_rms**2 / math.pow(10, SNR_DB/10))) 
        #square of data^2 / ratio of the desired signal power to the noise power in linear scale. It converts the SNR from decibels to a linear scale 
        #it gets the ratio of the desired signal power to the noise power in linear scale
        k = desired_noise_rms / background_noise_rms #scaling factor to apply to the bck noise to obtain the desired noise. 
        background_noise *= k

        # Apply RevEcho
        if(self.revecho):
            data, background_noise = self.revecho(data, background_noise)

        data = data + background_noise #adds the noise to the original background signal. 

        return data
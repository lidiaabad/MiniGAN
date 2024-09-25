import torch
import glob
import os
#from pesq import pesq
from joblib import Parallel, delayed
import numpy as np
import torch.nn as nn


def power_compress(x):
    # x[b, f_size, num_f] is a complex tensor (e.g., result of STFT)
    #real = x[..., 0]    # [ b, f_size, num_f]
    #imag = x[..., 1]    # [ b, f_size, num_f]´
    #spec = torch.complex(real, imag)
    
    # Magnitude and phase calculation
    mag = torch.abs(x)          # [b, f_size, num_f] - magnitude of the complex tensor
    phase = torch.angle(x)      # [b, f_size, num_f] - phase of the complex tensor
    
    # Nonlinear magnitude compression
    mag = mag ** 0.3            # Compress the magnitude
    
    # Reconstruct the compressed complex tensor
    real_compress = mag * torch.cos(phase)  # Recalculate the real part
    imag_compress = mag * torch.sin(phase)  # Recalculate the imaginary part
    
    return torch.stack([real_compress, imag_compress], 1)    # [ b, 2, f_size, num_f]

def power_uncompress(real, imag):
    nan_mask = torch.isnan(real)

        # Print the NaN elements
    if nan_mask.any():
        nan_elements = real[nan_mask]
        print('NaN elements in the tensor:', nan_elements)
        #torch.nan_to_num(est_mag)

    nan_mask2 = torch.isnan(imag)

        # Print the NaN elements
    if nan_mask2.any():
        nan_elements = imag[nan_mask2]
        print('NaN elements in the tensor:', nan_elements)
        #torch.nan_to_num(est_phase)
    
    spec = torch.complex(real, imag)

    nan_mask3 = torch.isnan(spec)

        # Print the NaN elements
    if nan_mask2.any():
        nan_elements = spec[nan_mask3]
        print('NaN elements in the tensor:', nan_elements)
        #torch.nan_to_num(est_phase)

    
    #zero_phase_mask = (torch.angle(spec) == 0.0)
    #real = torch.where(zero_phase_mask, real + 1e-9, real)
    #imag = torch.where(zero_phase_mask, imag + 1e-9, imag)

    # Re-create the complex tensor after adjustment
    #spec = torch.complex(real, imag)

    #torch.nan_to_num(spec)
    mag = torch.abs(spec)
    #torch.nan_to_num(mag)

    phase = torch.angle(spec)
    #phase.register_hook(lambda grad: grad.clamp(max=1e9))

    #torch.nan_to_num(phase)
    mag = mag**(1./0.3)
    #torch.nan_to_num(mag)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    #torch.nan_to_num(real_compress)
    #torch.nan_to_num(imag_compress)
    return torch.complex(real_compress, imag_compress)
    #return torch.stack([real_compress, imag_compress], -1)


def get_spec(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1./0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)


def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to('cuda')


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


class LearnableSigmoid_1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
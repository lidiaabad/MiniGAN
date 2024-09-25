import math
import torch
import torch.nn as nn
import numpy as np
#import onnxruntime as ort

from resample import *
import features as featurelib


class SerializableModule(nn.Module):

    subclasses = {}

    def __init__(self):
        super().__init__()

    @classmethod
    def register_model(cls, model_name):
        def decorator(subclass):
            cls.subclasses[model_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, arc):
        if arc not in cls.subclasses:
            raise ValueError('Bad model name {}'.format(arc))

        return cls.subclasses[arc]()

    def save(self, filename):
        torch.save(self.state_dict(), filename +'.pt')

    def save_entire_model(self, filename):
        torch.save(self, filename +'_entire.pt')

    def save_scripted(self, filename):
        scripted_module = torch.jit.script(self)
        scripted_module.save(filename + '.jit')

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


class ResBlock1d(nn.Module):
    def __init__(self, in_dim, internal_dim, k=3, s=1, p=1):
        super(ResBlock1d, self).__init__()
        
        # Use 2 ConvBlock in 1 ResBlock
        conv_block_1 = ConvBlock1d(in_dim, internal_dim, k=k, s=s, p=p, non_linear='relu')
        conv_block_2 = ConvBlock1d(internal_dim, in_dim, k=k, s=s, p=p, non_linear=None)
        self.res_block = nn.Sequential(conv_block_1, conv_block_2)
    
    def forward(self, x):
        out = x + self.res_block(x)
        return out


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, non_linear='elu'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        self.norm = nn.BatchNorm1d(num_features=out_channels)
        if(non_linear == 'elu'):
            self.activation = nn.ELU()
        elif(non_linear == 'relu'):
            self.activation = nn.ReLU()
        else:
            self.activation = None

    def forward(self, x):
        """
        1D Causal convolution.
        Args:
            x: [B, C, T]
        Returns:
            [B, C, T]
        """
        x = self.conv(x)
        x = self.norm(x)
        if(self.activation):
            x = self.activation(x)
        return x


class TransConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, non_linear='elu'):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, output_padding=0)
        self.norm = nn.BatchNorm1d(num_features=out_channels)
        if(non_linear == 'elu'):
            self.activation = nn.ELU()
        elif(non_linear == 'relu'):
            self.activation = nn.ReLU()
        else:
            self.activation = None

    def forward(self, x):
        """
        1D Causal convolution.
        Args:
            x: [B, C, T]
        Returns:
            [B, C, T]
        """
        x = self.conv(x)
        x = self.norm(x)
        if(self.activation):
            x = self.activation(x)
        return x


@SerializableModule.register_model('crn')
class CRN(SerializableModule):
    """
    Input: [batch size, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.conv_block_1 = ConvBlock1d(1, 16, k=7, s=1, p=3)
        self.conv_block_2 = ConvBlock1d(16, 32, k=4, s=2, p=1)
        self.conv_block_3 = ConvBlock1d(32, 64, k=4, s=2, p=1)
        self.conv_block_4 = ConvBlock1d(64, 128, k=4, s=2, p=1)
        self.conv_block_5 = ConvBlock1d(128, 256, k=4, s=2, p=1)

        # LSTM
        self.lstm_layer = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)

        # Decoder
        self.tran_conv_block_1 = TransConvBlock1d(256 + 256, 128, k=4, s=2, p=1)
        self.tran_conv_block_2 = TransConvBlock1d(128 + 128, 64, k=4, s=2, p=1)
        self.tran_conv_block_3 = TransConvBlock1d(64 + 64, 32, k=4, s=2, p=1)
        self.tran_conv_block_4 = TransConvBlock1d(32 + 32, 16, k=4, s=2, p=1)
        self.tran_conv_block_5 = TransConvBlock1d(16 + 16, 1, k=7, s=1, p=3, non_linear='none')

    def forward(self, x):

        # Pad to a valid dimension
        length = x.shape[1]
        x = nn.functional.pad(x, (0, self.valid_length(length) - length)) # [b, t]

        # Add channel dimension
        x = x.unsqueeze(1) # [b, 1, 24000]

        self.lstm_layer.flatten_parameters()

        e_1 = self.conv_block_1(x) # [b, 16, 24000]
        e_2 = self.conv_block_2(e_1) # [b, 32, 12000]
        e_3 = self.conv_block_3(e_2) # [b, 64, 6000]
        e_4 = self.conv_block_4(e_3) # [b, 128, 3000]
        e_5 = self.conv_block_5(e_4)  # [b, 256, 1500]

        # [b, 256, 1500] => [b, 1500, 256]
        lstm_in = e_5.permute(0, 2, 1)
        lstm_out, _ = self.lstm_layer(lstm_in)  # [b, 1500, 256]
        lstm_out = lstm_out.permute(0, 2, 1) # [b, 256, 1500]

        d_1 = self.tran_conv_block_1(torch.cat((lstm_out, e_5), 1)) # [b, 128, 3000]
        d_2 = self.tran_conv_block_2(torch.cat((d_1, e_4), 1)) # [b, 64, 6000]
        d_3 = self.tran_conv_block_3(torch.cat((d_2, e_3), 1)) # [b, 32, 12000]
        d_4 = self.tran_conv_block_4(torch.cat((d_3, e_2), 1)) # [b, 16, 24000]
        d_5 = self.tran_conv_block_5(torch.cat((d_4, e_1), 1)) # [b, 1, 24000]

        # Remove channel dimension
        enhanced = d_5.squeeze(1) # [b, 24000]

        return enhanced

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model
        """
        k, s, p = 7, 1, 3
        length = math.ceil(((length + 2*p -k) / s) + 1)
        k, s, p = 4, 2, 1
        for idx in range(4):
            length = math.ceil(((length + 2*p -k) / s) + 1)
            length = max(length, 1)
        for idx in range(4):
            length = math.ceil((length - 1)*s - 2*p + k)
        k, s, p = 7, 1, 3
        length = math.ceil((length - 1)*s - 2*p + k)
        return int(length)


@SerializableModule.register_model('gruse')
class GRUSE(SerializableModule):
    """
    Input: [batch size, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.conv_block_1 = ConvBlock1d(1, 16, k=7, s=1, p=3)
        self.conv_block_2 = ConvBlock1d(16, 32, k=4, s=2, p=1)
        self.conv_block_3 = ConvBlock1d(32, 64, k=4, s=2, p=1)
        self.conv_block_4 = ConvBlock1d(64, 128, k=4, s=2, p=1)
        self.conv_block_5 = ConvBlock1d(128, 256, k=4, s=2, p=1)

        # GRU
        self.gru_layer = nn.GRU(input_size=256, hidden_size=256, num_layers=2, batch_first=True)

        # Decoder
        self.tran_conv_block_1 = TransConvBlock1d(256 + 256, 128, k=4, s=2, p=1)
        self.tran_conv_block_2 = TransConvBlock1d(128 + 128, 64, k=4, s=2, p=1)
        self.tran_conv_block_3 = TransConvBlock1d(64 + 64, 32, k=4, s=2, p=1)
        self.tran_conv_block_4 = TransConvBlock1d(32 + 32, 16, k=4, s=2, p=1)
        self.tran_conv_block_5 = TransConvBlock1d(16 + 16, 1, k=7, s=1, p=3, non_linear='none')

    def forward(self, x):

        # Pad to a valid dimension
        length = x.shape[1]
        x = nn.functional.pad(x, (0, self.valid_length(length) - length)) # [b, t]

        # Add channel dimension
        x = x.unsqueeze(1) # [b, 1, 24000]

        self.gru_layer.flatten_parameters()

        e_1 = self.conv_block_1(x) # [b, 16, 24000]
        e_2 = self.conv_block_2(e_1) # [b, 32, 12000]
        e_3 = self.conv_block_3(e_2) # [b, 64, 6000]
        e_4 = self.conv_block_4(e_3) # [b, 128, 3000]
        e_5 = self.conv_block_5(e_4)  # [b, 256, 1500]

        # [b, 256, 1500] => [b, 1500, 256]
        gru_in = e_5.permute(0, 2, 1)
        gru_out, _ = self.gru_layer(gru_in)  # [b, 1500, 256]
        gru_out = gru_out.permute(0, 2, 1) # [b, 256, 1500]

        d_1 = self.tran_conv_block_1(torch.cat((gru_out, e_5), 1)) # [b, 128, 3000]
        d_2 = self.tran_conv_block_2(torch.cat((d_1, e_4), 1)) # [b, 64, 6000]
        d_3 = self.tran_conv_block_3(torch.cat((d_2, e_3), 1)) # [b, 32, 12000]
        d_4 = self.tran_conv_block_4(torch.cat((d_3, e_2), 1)) # [b, 16, 24000]
        d_5 = self.tran_conv_block_5(torch.cat((d_4, e_1), 1)) # [b, 1, 24000]

        # Remove channel dimension
        enhanced = d_5.squeeze(1) # [b, 24000]

        return enhanced

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model
        """
        k, s, p = 7, 1, 3
        length = math.ceil(((length + 2*p -k) / s) + 1)
        k, s, p = 4, 2, 1
        for idx in range(4):
            length = math.ceil(((length + 2*p -k) / s) + 1)
            length = max(length, 1)
        for idx in range(4):
            length = math.ceil((length - 1)*s - 2*p + k)
        k, s, p = 7, 1, 3
        length = math.ceil((length - 1)*s - 2*p + k)
        return int(length)


@SerializableModule.register_model('demucs')
class Demucs(SerializableModule):
    """
    Demucs speech enhancement model.
    Implemented as done here: https://github.com/facebookresearch/denoiser
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
    """

    def __init__(self, chin=1, chout=1, hidden=48, depth=5, kernel_size=8, stride=4, causal=True,
        resample=4, growth=2, max_hidden=10_000, normalize=True, glu=True, rescale=0.1, floor=1e-3):

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            self.rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = nn.functional.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return (std * x).squeeze(1)

    def rescale_conv(self, conv, reference):
        std = conv.weight.std().detach()
        scale = (std / reference)**0.5
        conv.weight.data /= scale
        if conv.bias is not None:
            conv.bias.data /= scale

    def rescale_module(self, module, reference):
        for sub in module.modules():
            if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
                self.rescale_conv(sub, reference)


@SerializableModule.register_model('denoiser64')
class Denoiser64(SerializableModule):
    def __init__(self):
        super().__init__()
        self.model = Demucs(hidden=64)

    def forward(self, x):
        return self.model(x)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


@SerializableModule.register_model('denoiser48')
class Denoiser48(SerializableModule):
    def __init__(self):
        super().__init__()
        self.model = Demucs(hidden=48)

    def forward(self, x):
        return self.model(x)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


class NSnet2Enhancer(object):
    """NSnet2 enhancer class."""

    def __init__(self, model_path, cfg=None):
        """Instantiate NSnet2 given a trained model path."""
        self.cfg = {
            'winlen'   : 0.02,
            'hopfrac'  : 0.5,
            'fs'       : 16000,
            'mingain'  : -80,
            'feattype' : 'LogPow'
        }
        self.frameShift = float(self.cfg['winlen'])* float(self.cfg["hopfrac"])
        self.fs = int(self.cfg['fs'])
        self.Nfft = int(float(self.cfg['winlen'])*self.fs)
        self.mingain = 10**(self.cfg['mingain']/20)

        # Frequency bins
        self.frequency_bins = 161
        
        """load onnx model"""
        self.ort = ort.InferenceSession(model_path)
        self.dtype = np.float32

    def enhance(self, x):
        """Obtain the estimated filter"""
        onnx_inputs = {self.ort.get_inputs()[0].name: x.astype(self.dtype)}
        out = self.ort.run(None, onnx_inputs)
        return out[0]

    def to(self, string):
        """Torch models compatibility"""
        pass

    def eval(self):
        """Torch models compatibility"""
        pass

    def __call__(self, x):
        """Enhance a batch of Audio signals."""

        # From torch to numpy
        x = x.cpu().numpy()

        # Create needed arrays
        time_bins = math.ceil((x[0].shape[0]-int(self.cfg['winlen']*self.cfg['fs']))/int(self.frameShift*self.cfg['fs'])) + 2
        features_shape = (x.shape[0], ) + (time_bins, self.frequency_bins)
        batch_features = np.zeros(features_shape)
        spectrum_shape = (x.shape[0], ) + (self.frequency_bins, time_bins)
        batch_spectrum = np.zeros(spectrum_shape, dtype=np.complex128)

        # Get features for each audio signal
        for i in range(x.shape[0]):
            spectrum = featurelib.calcSpec(x[i], self.cfg)
            features = featurelib.calcFeat(spectrum, self.cfg)
            features = np.transpose(features)
            batch_spectrum[i,:,:] = spectrum
            batch_features[i,:,:] = features
        
        # Forward. Data shape: [batch x time x freq]
        out = self.enhance(batch_features)
        
        # Get output in the time domain
        raw_out = np.zeros(x.shape)
        for i in range(out.shape[0]):

            # limit suppression gain
            gain = np.transpose(out[i])
            gain = np.clip(gain, a_min=self.mingain, a_max=1.0)
            enhanced_spectrum = batch_spectrum[i]*gain

            # go back to time domain
            enhanced = featurelib.spec2sig(enhanced_spectrum, self.cfg)
            raw_out[i,:] = enhanced[-x[i].shape[0]:]

        return torch.from_numpy(raw_out).float()


@SerializableModule.register_model('resse')
class ResSE(SerializableModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock1d(1, 64, k=7, s=1, p=3, non_linear='relu'), #maintains input shape and creates 64 channels
            ConvBlock1d(64, 96, k=4, s=2, p=1, non_linear='relu'),
            ConvBlock1d(96, 128, k=4, s=2, p=1, non_linear='relu'),
            ConvBlock1d(128, 192, k=4, s=2, p=1, non_linear='relu'),
            ConvBlock1d(192, 256, k=4, s=2, p=1, non_linear='relu'),
            ConvBlock1d(256, 256, k=4, s=2, p=1, non_linear='relu'),
        )
        
        self.residuals = nn.Sequential(
            ResBlock1d(256,256),
            ResBlock1d(256,256),
            ResBlock1d(256,256)
        )
        self.decoder = nn.Sequential(
            TransConvBlock1d(256, 256, k=4, s=2, p=1, non_linear='relu'),
            TransConvBlock1d(256, 192, k=4, s=2, p=1, non_linear='relu'),
            TransConvBlock1d(192, 128, k=4, s=2, p=1, non_linear='relu'),
            TransConvBlock1d(128, 96, k=4, s=2, p=1, non_linear='relu'),
            TransConvBlock1d(96, 64, k=4, s=2, p=1, non_linear='relu'),
            ConvBlock1d(64, 1, k=7, s=1, p=3, non_linear='none') #Maintains input shape and combines into 1 channel
        )
        
    def forward(self, x):
        """
       +------------------------------+            +-----------------------------+          +----------------------------+
       |                              |            |                             |          |                            |
       |    Encoder                   |            |    ResBlocks                |          |  Decoder                   |
       |                              |            |                             |          |                            |
       |                              |            |                             |          |                            |
       |   +----+   +----+   +----+   |            |                             |          |   +----+   +----+   +----+ |
       |   |    |   |    |   |    |   |            |                             |          |   |    |   |    |   |    | |
       |   |    +-+-+    +-+-+    +-+--------------+                             +------------+-+    +-+-+    +-+-+    | |
       |   +----+ | +----+ | +----+ | |            |                             |          | | +----+ | +----+ | +----+ |
       |          |        |        | |            |                             |          | |        |        |        |
       |          |        |        | |            |                             |          | |        |        |        |
       |          |        |        | |            |                             |          | |        |        |        |
       |          |        |        | |            |                             |          | |        |        |        |
       +------------------------------+            +-----------------------------+          +----------------------------+
                  |        |        |---------------------------------------------------------|        |        |
                  |        +---------------------------------------------------------------------------+        |
                  +---------------------------------------------------------------------------------------------+
                                                           Skip connections
                                                       Ouputs from Decoder
                                                       should be added to the inputs
                                                       of the Decoder
        """

        # Pad to a valid dimension
        length = x.shape[1]
        x = nn.functional.pad(x, (0, self.valid_length(length) - length)) # [b, t]

        # Add channel dimension
        x = x.unsqueeze(1) # [b, 1, 24000]

        # Enconder
        encoder_outputs = []
        for e in self.encoder:
            x = e(x)
            encoder_outputs.append(x)

        # Resnet 
        x = self.residuals(x)

        # Decoder
        encoder_outputs.reverse()
        for i,d in enumerate(self.decoder):
            x = d(x+encoder_outputs[i])

        # Remove channel dimension
        x = x.squeeze(1) # [b, 24000]

        return x

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model
        """
        # Encoder
        k, s, p = 7, 1, 3
        length = math.ceil(((length + 2*p -k) / s) + 1)
        k, s, p = 4, 2, 1
        for idx in range(5):
            length = math.ceil(((length + 2*p -k) / s) + 1)
            length = max(length, 1)
        
        # Resnet
        k, s, p = 3, 1, 1
        for idx in range(6):
            length = math.ceil(((length + 2*p -k) / s) + 1)
            length = max(length, 1)
        
        # Decoder
        k, s, p = 4, 2, 1
        for idx in range(5):
            length = math.ceil((length - 1)*s - 2*p + k)
        k, s, p = 7, 1, 3
        length = math.ceil((length - 1)*s - 2*p + k)
        return int(length)


@SerializableModule.register_model('dummy')
class Dummy(SerializableModule):
    """
    Dummy model to test base PESQ and STOI
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


@SerializableModule.register_model('demucs_serializable')
class DemucsSerializable(SerializableModule):

    def __init__(self, chin=1, chout=1, hidden=48, depth=5, kernel_size=8, stride=4, causal=True,
        resample=4, growth=2, max_hidden=10_000, glu=True, rescale=0.1, floor=1e-3):

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        # Encoder and decoder
        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        # LSTM
        self.lstm = nn.LSTM(bidirectional=not causal, num_layers=2, hidden_size=chin, input_size=chin)

        # Rescale
        self.rescale_module(self, reference=rescale)

    def valid_length(self, length: int) -> (int):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    def rescale_conv(self, conv, reference):
        std = conv.weight.std().detach()
        scale = (std / reference)**0.5
        conv.weight.data /= scale
        if conv.bias is not None:
            conv.bias.data /= scale

    def rescale_module(self, module, reference):
        for sub in module.modules():
            if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
                self.rescale_conv(sub, reference)

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        # Normalize always
        mono = mix.mean(dim=1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True)
        mix = mix / (self.floor + std)
        
        length = mix.shape[-1]
        x = mix
        x = nn.functional.pad(x, (0, self.valid_length(length) - length))
        
        # Upsample: 4 factor
        x = upsample2(x)
        x = upsample2(x)

        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)

        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)

        # Downsample: 4 factor
        x = downsample2(x)
        x = downsample2(x)

        x = x[..., :length]
        return (std * x).squeeze(1)
    
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import random
import numpy as np
from utils import dsp
import random
import math
import torch
import torch.nn as nn

"""
Data augment adapted from denoiser project 
by Wiliam Fernando López Gavilánez
to be applied to a single audio inputs.
"""

class RevEcho():
    """
    Hacky Reverb but runs on GPU without slowing down training.
    This reverb adds a succession of attenuated echos of the input
    signal to itself. Intuitively, the delay of the first echo will happen
    after roughly 2x the radius of the room and is controlled by `first_delay`.
    Then RevEcho keeps adding echos with the same delay and further attenuation
    until the amplitude ratio between the last and first echo is 1e-3.
    The attenuation factor and the number of echos to adds is controlled
    by RT60 (measured in seconds). RT60 is the average time to get to -60dB
    (remember volume is measured over the squared amplitude so this matches
    the 1e-3 ratio).
    At each call to RevEcho, `first_delay`, `initial` and `RT60` are
    sampled from their range. Then, to prevent this reverb from being too regular,
    the delay time is resampled uniformly within `first_delay +- 10%`,
    as controlled by the `jitter` parameter. Finally, for a denser reverb,
    multiple trains of echos are added with different jitter noises.
    Args:
        - initial: amplitude of the first echo as a fraction
            of the input signal. For each sample, actually sampled from
            `[0, initial]`. Larger values means louder reverb. Physically,
            this would depend on the absorption of the room walls.
        - rt60: range of values to sample the RT60 in seconds, i.e.
            after RT60 seconds, the echo amplitude is 1e-3 of the first echo.
            The default values follow the recommendations of
            https://arxiv.org/ftp/arxiv/papers/2001/2001.08662.pdf, Section 2.4.
            Physically this would also be related to the absorption of the
            room walls and there is likely a relation between `RT60` and
            `initial`, which we ignore here.
        - first_delay: range of values to sample the first echo delay in seconds.
            The default values are equivalent to sampling a room of 3 to 10 meters.
        - repeat: how many train of echos with differents jitters to add.
            Higher values means a denser reverb.
        - jitter: jitter used to make each repetition of the reverb echo train
            slightly different. For instance a jitter of 0.1 means
            the delay between two echos will be in the range `first_delay +- 10%`,
            with the jittering noise being resampled after each single echo.
        - keep_clean: fraction of the reverb of the clean speech to add back
            to the ground truth. 0 = dereverberation, 1 = no dereverberation.
        - sample_rate: sample rate of the input signals.
    """

    def __init__(self, proba=0.5, initial=0.3, rt60=(0.3, 1.3), first_delay=(0.01, 0.03),
                 repeat=3, jitter=0.1, keep_clean=0.1, sample_rate=16000):
        super().__init__()
        self.proba = proba
        self.initial = initial
        self.rt60 = rt60
        self.first_delay = first_delay
        self.repeat = repeat
        self.jitter = jitter
        self.keep_clean = keep_clean
        self.sample_rate = sample_rate

    def _reverb(self, source, initial, first_delay, rt60):
        """
        Return the reverb for a single source.
        """
        length = source.shape[-1] #get shape
        reverb = np.zeros_like(source) #zeros in the form
        reverb = reverb.astype(np.float64)
        for _ in range(self.repeat):
            frac = 1  # what fraction of the first echo amplitude is still here
            echo = initial * source #first echo = initial amplitude * source signal
            while frac > 1e-3:
                # First jitter noise for the delay
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                delay = min(1 + int(jitter * first_delay * self.sample_rate),length)
                # Delay the echo in time by padding with zero on the left
                echo = np.pad(echo[:-delay], (delay, 0), 'constant') 
                #reverb = reverb.astype(np.int64)
                reverb += echo #IMP reverb is the output
                # Second jitter noise for the attenuation
                jitter = 1 + self.jitter * random.uniform(-1, 1)
                # we want, with `d` the attenuation, d**(rt60 / first_ms) = 1e-3
                # i.e. log10(d) = -3 * first_ms / rt60, so that
                attenuation = 10**(-3 * jitter * first_delay / rt60)
                echo *= attenuation
                frac *= attenuation
        return reverb

    def __call__(self, clean, noise):
        if random.random() >= self.proba: #if the random number is above the probability of applying rev, no rev
            return clean, noise
        
        # Sample characteristics for the reverb
        initial = random.random() * self.initial #random number bw 0 and initial (obv initial amplitude sould be the higher)
        first_delay = random.uniform(*self.first_delay) #random number in the interval which is given by first delay var
        rt60 = random.uniform(*self.rt60) #random number in the interval given by rt60 var

        #obtain reverberation for the noise signal and clean signal. 
        reverb_noise = self._reverb(noise, initial, first_delay, rt60)
        reverb_clean = self._reverb(clean, initial, first_delay, rt60)
        
        #the reverb of noise always added to the noise
        noise += reverb_noise
        #the reverb of clean divided into clean and noise. 
        #The lower the keep_clean, the less reverb added to the clean signal
        clean=clean.astype(np.float64)
        clean += self.keep_clean * reverb_clean
        noise += (1 - self.keep_clean) * reverb_clean

        return clean, noise


class BandMask():
    """BandMask.
    Maskes bands of frequencies. Similar to Park, Daniel S., et al.
    "Specaugment: A simple data augmentation method for automatic speech recognition."
    (https://arxiv.org/pdf/1904.08779.pdf) but over the waveform.
    """

    def __init__(self, maxwidth=0.2, bands=120, sample_rate=16000):
        """__init__.
        :param maxwidth: the maximum width to remove
        :param bands: number of bands
        :param sample_rate: signal sample rate
        """
        super().__init__()
        self.maxwidth = maxwidth
        self.bands = bands
        self.sample_rate = sample_rate

    def __call__(self, signal):
        '''
        uses LowPassFilter to obtain low and midlow filters from the signal. Then if foccusses on the low and removes the midlow
        the lowpassfilter is applied using as cuffof frequencies the low and high

        '''
        bandwidth = int(abs(self.maxwidth) * self.bands) #multiply bandwidth by number of bands
        #generates a mel freq array of self.bands elements, qith lower freq=40 and higher freq = sample rate/2
        #there freq are divided by the sample rate as normalization 
        mels = dsp.mel_frequencies(self.bands, 40, self.sample_rate/2) / self.sample_rate 
        #generates randomly low and high values that be used on the mel frequencies to select the min and max of cutoffs
        low = random.randrange(self.bands)
        high = random.randrange(low, min(self.bands, low + bandwidth)) 

        # Sinc Low Pass filters
        lpf = dsp.LowPassFilters([mels[low], mels[high]])
        low, midlow = lpf(signal)

        # Stop band filtering (~broad noth filter)
        out = signal - midlow + low #we eliminate the midlow and foccus on low freq of singnals
        return out

class WhiteNoise(nn.Module):
    """Additive White Gaussian (Normal distribution) Noise"""
    def __init__(self, min_snr_db=0, max_snr_db=20, p=0.5):
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p
    def forward(self, x):
        if self.p < random.random():
            return x
        snr_db = np.random.uniform(self.min_snr_db, self.max_snr_db)
        awgn = np.random.randn(*x.shape)
        awgn_rms = np.std(awgn)
        x_rms = np.sqrt(np.mean(x**2))
        desired_awgn_rms = np.sqrt(x_rms**2 / math.pow(10, snr_db / 10))
        k = desired_awgn_rms / awgn_rms
        x = x + k * awgn
        return x / np.max(np.abs(x))
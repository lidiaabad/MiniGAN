import numpy as np
from scipy.io import wavfile
from scipy.linalg import toeplitz, norm
from scipy.fftpack import fft
import math
from scipy import signal
from pesq import pesq
from pystoi import stoi
import librosa
from scipy.stats import mode
from joblib import Parallel, delayed


def compute_metrics(cleanFile, enhancedFile, Fs, path):
    alpha = 0.95


    data1 = cleanFile
    data2 = enhancedFile
    sampling_rate1 = Fs
    sampling_rate2 = Fs

    if len(data1) != len(data2):
        length = min(len(data1), len(data2))
        data1 = data1[0:length] + np.spacing(1)
        data2 = data2[0:length] + np.spacing(1)

    # Compute metrics in parallel
    pesq_mos, wss_dist, llr_mean, snr_dist, STOI = Parallel(n_jobs=-1)(
        delayed(metric)(data1, data2, sampling_rate1) for metric in [get_pesq, wss, llr, snr, get_stoi]
    )
    # now compute the composite measures
    CSIG = 3.093 - 1.029 * llr_mean + 0.603 * pesq_mos - 0.009 * wss_dist
    CSIG = np.clip(CSIG, 1, 5)  # limit values to [1, 5]

    CBAK = 1.634 + 0.478 * pesq_mos - 0.007 * wss_dist + 0.063 * snr_dist
    CBAK = np.clip(CBAK, 1, 5)  # limit values to [1, 5]

    COVL = 1.594 + 0.805 * pesq_mos - 0.512 * llr_mean - 0.007 * wss_dist
    COVL = np.clip(COVL, 1, 5)  # limit values to [1, 5]

    STOI = get_stoi(data1, data2)


    return pesq_mos, CSIG, CBAK, COVL, snr_dist, STOI


def wss(clean_speech, processed_speech, sample_rate):
    # Compute Short-Time Fourier Transform (STFT) for clean and processed speech
    clean_stft = librosa.stft(clean_speech)
    processed_stft = librosa.stft(processed_speech)

    # Compute magnitude spectra
    clean_mag = np.abs(clean_stft)
    processed_mag = np.abs(processed_stft)

    # Compute spectral slopes
    clean_slope = np.diff(clean_mag, axis=1)
    processed_slope = np.diff(processed_mag, axis=1)

    # Compute WSS measure
    distortion = np.mean((clean_slope - processed_slope) ** 2)
    return distortion


def llr(clean_speech, processed_speech, sample_rate):
    # Compute window length and skip rate
    winlength = int(np.round(30 * sample_rate / 1000))
    hop_length = int(np.floor(winlength / 4))

    # Compute LPC Analysis Order
    order = 10 if sample_rate < 10000 else 16

    # Compute LPC coefficients for clean and processed speech
    clean_lpc = librosa.lpc(clean_speech, order=order, axis=0)
    processed_lpc = librosa.lpc(processed_speech, order=order, axis=0)

    # Compute Log Likelihood Ratio (LLR) measure
    numerator = np.dot(clean_lpc.T, clean_lpc)
    denominator = np.dot(processed_lpc.T, processed_lpc)
    distortion = np.log(numerator / denominator)
    mode_value, _ = mode(distortion[~np.isnan(distortion)])
    return mode_value

def snr(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech. Must be the same.
    clean_length = len(clean_speech)
    processed_length = len(processed_speech)
    if clean_length != processed_length:
        raise ValueError("Both Speech Files must be same length.")

    # Compute overall SNR
    overall_snr = 10 * np.log10(np.sum(np.square(clean_speech)) / np.sum(np.square(clean_speech - processed_speech)))
    return overall_snr

def get_pesq(ref_sig, out_sig, sr=16000):
    pesq_val = 0
    # To give fault tolerance as the pesq library has this bug
    try:
        for i in range(len(ref_sig)):
            pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')
    except:
        return 0
    return pesq_val


def get_stoi(ref_sig, out_sig, sr=16000):

    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val

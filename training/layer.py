from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils
import numpy as np
import math
import tensorflow as tf

from keras import initializers




class MusicSinc1D(Layer):
    def __init__(self, N_filt=256, Filt_dim=2501, fs=22050):
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

        super(MusicSinc1D, self).__init__()
        # The filters are trainable parameters.
        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = 2595 * np.log10(1 + (self.fs / 2) / 700)  # Convert Hz to Mel
        mel_points = np.linspace(
            low_freq_mel, high_freq_mel, self.N_filt
        )  # Equally spaced in Mel scale
        f_cos = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (self.fs / 2) - 100
        self.freq_scale = self.fs * 10
        self.filt_b1 = K.variable(b1 / self.freq_scale)
        self.filt_band = K.variable((b2 - b1) / self.freq_scale)

        # Get beginning and end frequencies of the filters.
        min_freq = 50.0
        min_band = 50.0
        self.filt_beg_freq = K.abs(self.filt_b1) + min_freq / self.freq_scale
        self.filt_end_freq = self.filt_beg_freq + (
            K.abs(self.filt_band) + min_band / self.freq_scale
        )

        # Filter window (hamming).
        n = np.linspace(0, self.Filt_dim, self.Filt_dim)
        window = 0.54 - 0.46 * K.cos(2 * math.pi * n / self.Filt_dim)
        window = K.cast(window, "float32")
        self.window = K.variable(window)

        # TODO what is this?
        t_right_linspace = np.linspace(
            1, (self.Filt_dim - 1) / 2, int((self.Filt_dim - 1) / 2)
        )
        self.t_right = K.variable(t_right_linspace / self.fs)



        # Compute the filters.
        output_list = []
        for i in range(self.N_filt):
            low_pass1 = (
                2
                * self.filt_beg_freq[i]
                * sinc(self.filt_beg_freq[i] * self.freq_scale, self.t_right)
            )
            low_pass2 = (
                2
                * self.filt_end_freq[i]
                * sinc(self.filt_end_freq[i] * self.freq_scale, self.t_right)
            )
            band_pass = low_pass2 - low_pass1
            band_pass = band_pass / K.max(band_pass)
            output_list.append(band_pass * self.window)
        self.filters = K.stack(output_list)  # (80, 251)
        self.filters = K.transpose(self.filters)  # (251, 80)
        self.filters = K.reshape(
            self.filters, (self.Filt_dim, 1, self.N_filt)
        )  # (251,1,80) in TF: (filter_width, in_channels, out_channels) in
        # PyTorch (out_channels, in_channels, filter_width)
        # Do the convolution.
    def call(self, x):
        out = K.conv1d(x, kernel=self.filters)

        return out

'''
    def compute_output_shape(self, input_shape):
        new_size = conv_utils.conv_output_length(
            input_shape[1], self.Filt_dim, padding="valid", stride=1, dilation=1
        )
        return (input_shape[0],) + (new_size,) + (self.N_filt,)
'''

def sinc(band, t_right):
    y_right = K.sin(2 * band * t_right) / (2 * band * t_right)
    # y_left = flip(y_right, 0) TODO remove if useless
    y_left = K.reverse(y_right, 0)
    y = K.concatenate([y_left, K.variable(K.ones(1)), y_right])
    return y

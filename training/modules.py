import numpy as np
import tensorflow as tf
import math
import librosa
from kapre import STFT, Magnitude, MagnitudeToDecibel
from tensorflow.keras.layers import (
    Conv1D,
    MaxPool1D,
    BatchNormalization,
    GlobalAvgPool1D,
    Multiply,
    GlobalMaxPool1D,
    Dense,
    Dropout,
    Activation,
    Reshape,
    Input,
    Concatenate,
    Add,
    ZeroPadding1D,
    LeakyReLU,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils

from keras import initializers


def hz_to_midi(hz):
    return 12 * (torch.log2(hz) - np.log2(440.0)) + 69


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def note_to_hz(note):
    return librosa.core.note_to_hz(note)


def note_to_midi(note):
    return librosa.core.note_to_midi(note)


def hz_to_note(hz):
    return librosa.core.hz_to_note(hz)


def initialize_filterbank(sample_rate, n_harmonic, semitone_scale):
    # MIDI
    # lowest note
    low_midi = note_to_midi("C1")

    # highest note
    high_note = hz_to_note(sample_rate / (2 * n_harmonic))
    high_midi = note_to_midi(high_note)

    # number of scales
    level = (high_midi - low_midi) * semitone_scale
    midi = np.linspace(low_midi, high_midi, level + 1)
    # midi = tf.cast(midi, tf.float32)
    hz = midi_to_hz(midi[:-1])

    # stack harmonics
    harmonic_hz = []
    for i in range(n_harmonic):
        harmonic_hz = np.concatenate((harmonic_hz, hz * (i + 1)))

    return harmonic_hz, level


class HarmonicSTFT(tf.keras.layers.Layer):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=None,
        hop_length=None,
        pad=0,
        power=2,
        normalized=False,
        n_harmonic=6,
        semitone_scale=2,
        bw_Q=1.0,
        learn_bw=None,
    ):
        super(HarmonicSTFT, self).__init__()

        # Parameters
        self.sample_rate = sample_rate
        self.n_harmonic = n_harmonic
        self.bw_alpha = 0.1079
        self.bw_beta = 24.7
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fft_bins = tf.linspace(0, self.sample_rate // 2, self.n_fft // 2 + 1)
        self.stft = STFT(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=256,
            window_name="hann_window",
            input_shape=(80000, 1),
            pad_begin=True,
        )
        self.magnitude = Magnitude()
        self.to_decibel = MagnitudeToDecibel()
        self.zero = tf.zeros(
            [
                1,
            ]
        )
        # Spectrogram

        # Initialize the filterbank. Equally spaced in MIDI scale.
        harmonic_hz, self.level = initialize_filterbank(
            sample_rate, n_harmonic, semitone_scale
        )

        # Center frequncies to tensor
        self.f0 = tf.constant(harmonic_hz, dtype="float32")
        # Bandwidth parameters
        self.bw_Q = tf.Variable(np.array([bw_Q]), dtype="float32", trainable=True)

    def get_harmonic_fb(self):
        # bandwidth
        bw = (self.bw_alpha * self.f0 + self.bw_beta) / self.bw_Q
        bw = tf.dtypes.cast(tf.expand_dims(bw, axis=0), dtype=tf.float32)
        f0 = tf.dtypes.cast(tf.expand_dims(self.f0, axis=0), dtype=tf.float32)
        fft_bins = tf.dtypes.cast(
            tf.expand_dims(self.fft_bins, axis=1), dtype=tf.float32
        )

        up_slope = tf.matmul(fft_bins, (2 / bw)) + 1 - (2 * f0 / bw)
        down_slope = tf.matmul(fft_bins, (-2 / bw)) + 1 + (2 * f0 / bw)
        fb = tf.math.maximum(self.zero, tf.math.minimum(down_slope, up_slope))
        return fb

        # Scale magnitude relative to maximum value in S. Zeros in the output
        # correspond to positions where S == ref.
        ref = tf.reduce_max(S)

        log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
        log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

        return log_spec

    def call(self, input_tensor, training=False):
        # 80000, 1
        waveform = tf.keras.layers.Reshape((-1, 1))(input_tensor)

        # stft
        spec = self.stft(waveform)
        spec = self.magnitude(spec)

        harmonic_fb = self.get_harmonic_fb()
        harmonic_fb = tf.expand_dims(harmonic_fb, axis=0)
        harmonic_spec = tf.matmul(tf.transpose(spec, perm=[0, 3, 1, 2]), harmonic_fb)
        b, c, w, h = harmonic_spec.shape
        harmonic_spec = tf.keras.layers.Reshape(
            (-1, h // self.n_harmonic, self.n_harmonic)
        )(harmonic_spec)
        harmonic_spec = tf.transpose(harmonic_spec, perm=[0, 2, 1, 3])
        harmonic_spec = self.to_decibel(harmonic_spec)
        return harmonic_spec


class ResNet_mtat(tf.keras.layers.Layer):
    def __init__(self, input_channels, conv_channels=128):
        super(ResNet_mtat, self).__init__()
        self.num_class = 50

        # residual convolution
        self.res1 = Conv3_2d(conv_channels, 2)
        self.res2 = Conv3_2d_resmp(conv_channels, 2)
        self.res3 = Conv3_2d_resmp(conv_channels, 2)
        self.res4 = Conv3_2d_resmp(conv_channels, 2)
        self.res5 = Conv3_2d(conv_channels * 2, 2)
        self.res6 = Conv3_2d_resmp(conv_channels * 2, (2, 3))
        self.res7 = Conv3_2d_resmp(conv_channels * 2, (2, 3))

        # fully connected
        self.fc_1 = tf.keras.layers.Dense(conv_channels * 2)
        self.bn = tf.keras.layers.BatchNormalization()
        self.fc_2 = tf.keras.layers.Dense(self.num_class)
        self.activation = tf.keras.layers.Activation("sigmoid")
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.relu = tf.keras.layers.Activation("relu")
        self.gmp = tf.keras.layers.GlobalMaxPooling2D()

    def call(self, x, training=False):
        # residual convolution
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        x = self.res3(x, training=training)
        x = self.res4(x, training=training)
        x = self.res5(x, training=training)
        x = self.res6(x, training=training)
        x = self.res7(x, training=training)

        # global max pooling
        # (16, 256)
        x = self.gmp(x)

        # fully connected
        x = self.fc_1(x, training=training)
        x = self.bn(x, training=training)
        x = self.relu(x, training=training)
        x = self.dropout(x, training=training)
        x = self.fc_2(x, training=training)
        x = self.activation(x, training=training)
        return x


############################################################
# modules for joint embeddings
#############################################################


class DenseLeakyReluLayer(tf.keras.layers.Layer):
    """A dense layer followed by a LeakyRelu layer"""

    def __init__(self, n, alpha=0.3):
        super(DenseLeakyReluLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(n, activation=None)
        self.norm = tf.keras.layers.BatchNormalization()
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=alpha)

    def call(self, input_tensor):
        x = self.dense(input_tensor)
        return self.lrelu(x)


class UnitNormLayer(tf.keras.layers.Layer):
    """Normalize vectors (euclidean norm) in batch to unit hypersphere."""

    def __init__(self):
        super(UnitNormLayer, self).__init__()

    def call(self, input_tensor):
        norm = tf.norm(input_tensor, axis=1)
        return input_tensor / tf.reshape(norm, [-1, 1])


class Wave_ResNet(tf.keras.layers.Layer):
    def __init__(self, input_channels, conv_channels=128, dataset="mtat"):
        super(Wave_ResNet, self).__init__()
        if dataset == "mtat":
            self.num_class = 50
        elif dataset == "dcase":
            self.num_class = 17
        elif dataset == "keyword":
            self.num_class = 35

        self.res1 = Conv3_2d(conv_channels, 2)
        self.res2 = Conv3_2d_resmp(conv_channels, 2)
        self.res3 = Conv3_2d_resmp(conv_channels, 2)
        self.res4 = Conv3_2d(conv_channels * 2, 2)
        self.res5 = Conv3_2d_resmp(conv_channels * 2, 2)
        self.res6 = Conv3_2d_resmp(conv_channels * 2, (2, 3))
        self.res7 = Conv3_2d(conv_channels * 2 * 2, (2, 3))
        # fully connected
        self.gmp = tf.keras.layers.GlobalMaxPooling2D()
        self.concat = tf.keras.layers.Concatenate()

    def call(self, x, training=False):
        # residual convolution
        x1 = self.res1(x, training=training)
        x2 = self.res2(x1, training=training)
        x3 = self.res3(x2, training=training)
        x4 = self.res4(x3, training=training)
        x5 = self.res5(x4, training=training)
        x6 = self.res6(x5, training=training)
        x7 = self.res7(x6, training=training)

        return self.concat([self.gmp(x5), self.gmp(x6), self.gmp(x7)])


class Conv3_2d(tf.keras.layers.Layer):
    def __init__(self, output_channels, pooling=2):
        super(Conv3_2d, self).__init__()
        self.conv = tf.keras.layers.Conv2D(
            output_channels, kernel_size=(3, 3), padding="same"
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation("relu")
        self.mp = tf.keras.layers.MaxPooling2D((pooling))

    def call(self, x, training=False):
        out = self.mp(
            self.relu(self.bn(self.conv(x, training=training), training=training)),
            training=training,
        )
        return out


class Conv3_2d_resmp(tf.keras.layers.Layer):
    def __init__(self, output_channels, pooling=2):
        super(Conv3_2d_resmp, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(
            output_channels, kernel_size=(3, 3), padding="same"
        )
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(
            output_channels, kernel_size=(3, 3), padding="same"
        )
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation("relu")
        self.mp = tf.keras.layers.MaxPooling2D((pooling))

    def call(self, x, training=False):
        out = self.bn_2(
            self.conv_2(
                self.relu(
                    self.bn_1(self.conv_1(x, training=training), training=training)
                ),
                training=training,
            ),
            training=training,
        )
        out = x + out
        out = self.mp(self.relu(out), training=training)
        return out


##########################################################
# Modules for resemul
#########################################################
def se_fn(x, amplifying_ratio):
    num_features = x.shape[-1]
    x = GlobalAvgPool1D()(x)
    x = Reshape((1, num_features))(x)
    x = Dense(
        num_features * amplifying_ratio,
        kernel_initializer="glorot_uniform",
    )(x)
    x = Activation("relu")(x)
    x = Dense(num_features, activation="sigmoid", kernel_initializer="glorot_uniform")(
        x
    )
    return x


def se_block(x, num_features, weight_decay, amplifying_ratio):
    x = basic_block(x, num_features, weight_decay, amplifying_ratio)
    x = Multiply()([x, se_fn(x, amplifying_ratio)])
    return x


def basic_block(x, num_features, weight_decay, _):
    x = Conv1D(
        num_features,
        kernel_size=3,
        padding="same",
        use_bias=True,
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool1D(pool_size=3)(x)
    return x


def rese_block(x, num_features, weight_decay, amplifying_ratio):
    if num_features != x.shape[-1]:
        shortcut = Conv1D(
            num_features,
            kernel_size=1,
            padding="same",
            use_bias=True,
            kernel_regularizer=l2(weight_decay),
            kernel_initializer="glorot_uniform",
        )(x)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = x
    x = Conv1D(
        num_features,
        kernel_size=3,
        padding="same",
        use_bias=True,
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)
    x = Conv1D(
        num_features,
        kernel_size=3,
        padding="same",
        use_bias=True,
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)
    x = BatchNormalization()(x)
    if amplifying_ratio > 0:
        x = Multiply()([x, se_fn(x, amplifying_ratio)])
    x = Add()([shortcut, x])
    x = Activation("relu")(x)
    x = MaxPool1D(pool_size=3)(x)
    return x


class MusicSinc1D(Layer):
    def __init__(
        self, N_filt=256, Filt_dim=2501, fs=16000, stride=1, padding="SAME", **kwargs
    ):
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.stride = stride
        self.padding = padding

        super(MusicSinc1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # The filters are trainable parameters.
        self.filt_b1 = self.add_weight(
            name="filt_b1",
            shape=(self.N_filt, 1),
            initializer="uniform",
            trainable=True,
        )
        self.filt_band = self.add_weight(
            name="filt_band",
            shape=(self.N_filt, 1),
            initializer="uniform",
            trainable=True,
        )

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
        self.B1 = np.expand_dims(b1, axis=-1)
        self.B2 = np.expand_dims(b2, axis=-1)
        self.freq_scale = self.fs * 1.0

        t_right = tf.constant(
            tf.linspace(1.0, (self.Filt_dim - 1) / 2, int((self.Filt_dim - 1) / 2))
            / self.fs,
            tf.float32,
        )
        self.T_Right = tf.tile(tf.expand_dims(t_right, axis=0), (self.N_filt, 1))

        n = tf.linspace(0, self.Filt_dim - 1, self.Filt_dim)
        window = 0.54 - 0.46 * tf.cos(2 * math.pi * n / self.Filt_dim)
        window = tf.cast(window, tf.float32)
        self.Window = tf.tile(tf.expand_dims(window, axis=0), (self.N_filt, 1))

        self.set_weights(
            [self.B1 / self.freq_scale, (self.B2 - self.B1) / self.freq_scale]
        )

        super(MusicSinc1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):

        min_freq = 50.0
        min_band = 50.0

        filt_beg_freq = tf.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (
            tf.abs(self.filt_band) + min_band / self.freq_scale
        )

        low_pass1 = 2 * filt_beg_freq * self.sinc(filt_beg_freq * self.freq_scale)
        low_pass2 = 2 * filt_end_freq * self.sinc(filt_end_freq * self.freq_scale)
        band_pass = low_pass2 - low_pass1
        band_pass = band_pass / tf.reduce_max(band_pass, axis=1, keepdims=True)
        windowed_band_pass = band_pass * self.Window

        filters = tf.transpose(windowed_band_pass)
        filters = tf.reshape(filters, (self.Filt_dim, 1, self.N_filt))

        # Do the convolution.
        out = tf.nn.conv1d(x, filters=filters, stride=self.stride, padding=self.padding)

        return out

    def sinc(self, band):
        y_right = tf.sin(2 * band * self.T_Right) / (2 * band * self.T_Right)
        y_left = tf.reverse(y_right, [1])
        y = tf.concat([y_left, tf.ones((self.N_filt, 1)), y_right], axis=1)
        return y

    def compute_output_shape(self, input_shape):
        new_size = conv_utils.conv_output_length(
            input_shape[1],
            self.Filt_dim,
            padding=self.padding.lower(),
            stride=self.stride,
            dilation=1,
        )
        return (input_shape[0],) + (new_size,) + (self.N_filt,)

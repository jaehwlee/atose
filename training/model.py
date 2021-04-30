# coding: utf-8
import numpy as np
import tensorflow as tf
from modules import HarmonicSTFT
from modules import DenseLeakyReluLayer, UnitNormLayer

class Model(tf.keras.Model):
    def __init__(self,
                conv_channels=128,
                sample_rate=16000,
                n_fft=512,
                n_harmonic=6,
                semitone_scale=2,
                learn_bw=None,
                dataset='mtat'):
        super(Model, self).__init__()

        self.hstft = HarmonicSTFT(sample_rate=sample_rate,
                                    n_fft=n_fft,
                                    n_harmonic=n_harmonic,
                                    semitone_scale=semitone_scale,
                                    learn_bw=learn_bw)
        self.hstft_bn = tf.keras.layers.BatchNormalization()

        # 2D CNN
        if dataset == 'mtat':
            from modules import ResNet_mtat as ResNet
        self.conv_2d = ResNet(input_channels=n_harmonic, conv_channels=conv_channels)

    def call(self, x, training=False):
        # harmonic stft
        x = self.hstft_bn(self.hstft(x, training=training), training=training)

        # 2D CNN
        logits = self.conv_2d(x, training=training)

        return logits



####################################################
# for tag ae
###################################################


class WaveEncoder(tf.keras.Model):
    def __init__(self,
                conv_channels=128,
                sample_rate=16000,
                n_fft=512,
                n_harmonic=6,
                semitone_scale=2,
                learn_bw=None,
                dataset='mtat'):
        super(WaveEncoder, self).__init__()

        self.hstft = HarmonicSTFT(sample_rate=sample_rate,
                                    n_fft=n_fft,
                                    n_harmonic=n_harmonic,
                                    semitone_scale=semitone_scale,
                                    learn_bw=learn_bw)
        self.hstft_bn = tf.keras.layers.BatchNormalization()

        # 2D CNN
        if dataset == 'mtat':
            from modules import Wave_ResNet as ResNet
        self.conv_2d = ResNet(input_channels=n_harmonic, conv_channels=conv_channels)

    def call(self, x, training=False):
        # harmonic stft
        x = self.hstft_bn(self.hstft(x, training=training), training=training)

        # 2D CNN
        logits = self.conv_2d(x, training=training)

        return logits



class WaveProjector(tf.keras.Model):
    """
    A projection network, P(·), which maps the normalized representation vector r into a vector z = P(r) ∈ R^{DP} 
    suitable for computation of the contrastive loss.
    """

    def __init__(self, n, normalize=True, activation="leaky_relu"):
        super(WaveProjector, self).__init__(name="")
        if activation == "leaky_relu":
            self.dense = DenseLeakyReluLayer(n)
            self.dense2 = DenseLeakyReluLayer(n)
        else:
            self.dense = tf.keras.layers.Dense(n, activation=activation)
            self.dense2 = tf.keras.layers.Dense(n, activation=activation)

        self.normalize = normalize
        if self.normalize:
            self.norm = UnitNormLayer()

    def call(self, input_tensor, training=False):
        x = self.dense(input_tensor, training=training)
        x = self.dense2(x, training=training)
        if self.normalize:
            x = self.norm(x)
        return x




class SupervisedClassifier(tf.keras.Model):
    """For stage 2, simply a softmax on top of the Encoder.
    """

    def __init__(self, num_classes=50):
        super(SupervisedClassifier, self).__init__(name="")
        self.dense = DenseLeakyReluLayer(256)
        self.dense2 = DenseLeakyReluLayer(256)
        self.normalize = UnitNormLayer()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(num_classes, activation="sigmoid")

    def call(self, input_tensor, training=False):
        x = self.dense(input_tensor, training=training)
        x = self.dense2(x,training=training)
        x = self.normalize(x)
        x = self.dropout(x, training=training)
        x = self.dense3(x, training=training)
        return x



class TagEncoder(tf.keras.Model):
    """An encoder network, E(·), which maps an augmented image x to a representation vector, r = E(x) ∈ R^{DE}
    """

    def __init__(self, n, normalize=True, activation="leaky_relu"):
        super(TagEncoder, self).__init__(name="")

        if activation == "leaky_relu":
            self.hidden1 = DenseLeakyReluLayer(n)
            self.hidden2 = DenseLeakyReluLayer(n)
            #self.hidden3 = DenseLeakyReluLayer(128)

        else:
            self.hidden1 = tf.keras.layers.Dense(n, activation=activation)
            self.hidden2 = tf.keras.layers.Dense(n, activation=activation)
            #self.hidden3 = tf.keras.layers.Dense(128, activation=activation)

        self.normalize = normalize
        if self.normalize:
            self.norm = UnitNormLayer()

    def call(self, input_tensor, training=False):
        x = self.hidden1(input_tensor, training=training)
        x = self.hidden2(x, training=training)
        #x = self.hidden3(x, training=training)

        if self.normalize:
            x = self.norm(x)
        return x




class TagDecoder(tf.keras.Model):
    def __init__(self, n, dimension=50):
        super(TagDecoder, self).__init__(name="")
        self.dense = DenseLeakyReluLayer(n)
        self.dense2 = tf.keras.layers.Dense(dimension, activation="sigmoid")

    def call(self, input_tensor, training=False):
        x = self.dense(input_tensor, training=training)
        x = self.dense2(x, training=training)
        return x


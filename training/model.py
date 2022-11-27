# coding: utf-8
import numpy as np
import tensorflow as tf
from modules import HarmonicSTFT
from modules import DenseLeakyReluLayer, UnitNormLayer
from modules import se_fn, se_block, basic_block, rese_block
from modules import MusicSinc1D


class HarmonicCNN(tf.keras.Model):
    def __init__(
        self,
        conv_channels=128,
        sample_rate=16000,
        n_fft=512,
        n_harmonic=6,
        semitone_scale=2,
        learn_bw=None,
        dataset="mtat",
    ):
        super(HarmonicCNN, self).__init__()

        self.hstft = HarmonicSTFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_harmonic=n_harmonic,
            semitone_scale=semitone_scale,
            learn_bw=learn_bw,
        )
        self.hstft_bn = tf.keras.layers.BatchNormalization()

        # 2D CNN
        from modules import Wave_ResNet as ResNet

        self.conv_2d = ResNet(
            input_channels=n_harmonic, conv_channels=conv_channels, dataset=dataset
        )

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

    def __init__(self, n=128, normalize=True, activation="leaky_relu"):
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
    """For stage 2, simply a softmax on top of the Encoder."""

    def __init__(self, dataset="mtat"):
        super(SupervisedClassifier, self).__init__(name="")
        self.dense = DenseLeakyReluLayer(256)
        self.dense2 = DenseLeakyReluLayer(256)
        self.normalize = UnitNormLayer()
        self.dropout = tf.keras.layers.Dropout(0.5)
        if dataset == "mtat":
            num_classes = 50
            activation = "sigmoid"
        elif dataset == "dcase":
            activation = "sigmoid"
            num_classes = 17
        elif dataset == "keyword":
            activation = "softmax"
            num_classes = 35
        self.dense3 = tf.keras.layers.Dense(num_classes, activation=activation)

    def call(self, input_tensor, training=False):
        x = self.dense(input_tensor, training=training)
        x = self.dense2(x, training=training)
        x = self.normalize(x)
        x = self.dropout(x, training=training)
        x = self.dense3(x, training=training)
        return x


class TagEncoder(tf.keras.Model):
    """An encoder network, E(·), which maps an augmented image x to a representation vector, r = E(x) ∈ R^{DE}"""

    def __init__(self, n=128, normalize=True, activation="leaky_relu"):
        super(TagEncoder, self).__init__(name="")

        self.hidden1 = DenseLeakyReluLayer(n)
        self.hidden2 = DenseLeakyReluLayer(n)

        self.normalize = normalize
        if self.normalize:
            self.norm = UnitNormLayer()

    def call(self, input_tensor, training=False):
        x = self.hidden1(input_tensor, training=training)
        x = self.hidden2(x, training=training)

        if self.normalize:
            x = self.norm(x)
        return x


class TagDecoder(tf.keras.Model):
    def __init__(self, n=128, dataset="mtat"):
        super(TagDecoder, self).__init__(name="")
        self.dense = DenseLeakyReluLayer(n)
        if dataset == "mtat":
            activation = "sigmoid"
            dimension = 50
        elif dataset == "dcase":
            activation = "sigmoid"
            dimension = 17
        elif dataset == "keyword":
            activation = "softmax"
            dimension = 35
        self.dense2 = tf.keras.layers.Dense(dimension, activation=activation)

    def call(self, input_tensor, training=False):
        x = self.dense(input_tensor, training=training)
        x = self.dense2(x, training=training)
        return x


def resemul(
    ms=False,
    block_type="se",
    amplifying_ratio=16,
    drop_rate=0.5,
    weight_decay=1e-4,
    num_classes=50,
    dataset="mtat",
):

    # Input&Reshape
    if block_type == "se":
        block = se_block
    elif block_type == "rese":
        block = rese_block
    elif block_type == "res":
        block = rese_block
        amplifying_ratio = -1
    elif block_type == "basic":
        block = basic_block
    inp = tf.keras.layers.Input(shape=(59049, 1))
    x = inp
    # Strided Conv
    num_features = 128
    if ms:
        if dataset == "mtat":
            filter_size = 2501
            filter_num = 256
            fs = 22050
        else:
            filter_size = 251
            filter_num = 128
            fs = 16000
        x = tf.keras.layers.ZeroPadding1D(padding=(filter_size - 1) // 2)(x)
        x = MusicSinc1D(filter_num, filter_size, fs)(x)

    x = tf.keras.layers.Conv1D(
        num_features,
        kernel_size=3,
        strides=3,
        padding="valid",
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        kernel_initializer="he_uniform",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # rese-block
    layer_outputs = []
    for i in range(9):
        num_features *= 2 if (i == 2 or i == 8) else 1
        x = block(x, num_features, weight_decay, amplifying_ratio)
        layer_outputs.append(x)

    x = tf.keras.layers.Concatenate()(
        [tf.keras.layers.GlobalMaxPool1D()(output) for output in layer_outputs[-3:]]
    )
    x = tf.keras.layers.Dense(x.shape[-1], kernel_initializer="glorot_uniform")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation("relu")(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model

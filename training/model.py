# coding: utf-8
import numpy as np
import tensorflow as tf
from modules import HarmonicSTFT
from modules import DenseLeakyReluLayer, UnitNormLayer
from modules import se_fn, se_block, basic_block, rese_block
from modules import MusicSinc1D

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
        #x = self.hidden3(x, training=training)

        if self.normalize:
            x = self.norm(x)
        return x




class TagDecoder(tf.keras.Model):
    def __init__(self, n=128, dimension=50):
        super(TagDecoder, self).__init__(name="")
        self.dense = DenseLeakyReluLayer(n)
        self.dense2 = tf.keras.layers.Dense(dimension, activation="sigmoid")

    def call(self, input_tensor, training=False):
        x = self.dense(input_tensor, training=training)
        x = self.dense2(x, training=training)
        return x


def resemul(
    ms=False, block_type="se", amplifying_ratio=16, drop_rate=0.5, weight_decay=1e-4, num_classes=50,
):

    # Input&Reshape
    if block_type == 'se':
        block = se_block
    elif block_type == 'rese':
        block = rese_block
    elif block_type == 'res':
        block = rese_block
        amplifying_ratio = -1
    elif block_type == 'basic':
        block = basic_block
    inp = tf.keras.layers.Input(shape=(59049, 1))
    x = inp
    # Strided Conv
    num_features = 128
    if ms:
        filter_size = 2501
        filter_num = 256
        x = tf.keras.layers.ZeroPadding1D(padding=(filter_size-1)//2)(x)
        x = MusicSinc1D(filter_num, filter_size, 22050)(x)
    
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
    #x = LeakyReLU(0.3)(x)
    x = tf.keras.layers.Activation("relu")(x)

    # rese-block
    layer_outputs = []
    for i in range(9):
        num_features *= 2 if (i == 2 or i == 8) else 1
        x = block(x, num_features, weight_decay, amplifying_ratio)
        layer_outputs.append(x)

    x = tf.keras.layers.Concatenate()([tf.keras.layers.GlobalMaxPool1D()(output) for output in layer_outputs[-3:]])
    x = tf.keras.layers.Dense(x.shape[-1], kernel_initializer="glorot_uniform")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #out = LeakyReLU(0.3)(x)
    out = tf.keras.layers.Activation("relu")(x)
    # if drop_rate > 0.0:
    #    x = Dropout(drop_rate)(x)
    # out = Dense(num_classes, activation="sigmoid", kernel_initializer="glorot_uniform")(
    #    x
    # )

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    return model


'''

class ReseMul(tf.keras.Model):
    def __init__(self, block_type='rese', amplifying_ratio=16, drop_rate=0.5, weight_decay=1e-4, num_classes=50):
        super(ReseMul, self).__init__(name="")
        self.amplifying_ratio = amplifying_ratio
        self.drop_rate = drop_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.num_features = num_features
        # Input&Reshape
        if block_type == 'se':
            self.block = se_block
        elif block_type == 'rese':
            self.block = rese_block
        elif block_type == 'res':
            self.block = rese_block
            self.amplifying_ratio = -1
        elif block_type == 'basic':
            self.block = basic_block

        self.conv1d_1 = Conv1D(self.num_features, kernel_size=3, padding="valid", use_bias=True, kernel_regularizer=l2(weight_decay), kernel_initializer="he_uniform")
        self.batchnorm_1 = BatchNormalization()
        self.activation = Activation("relu")
        

    def call(self, input_tensor, training=False):
        x = tf.keras.layers.Reshape((-1, 1))(input_tensor)
        num_features = 128
        x = self.conv1d_1(x, training=training)
        x = self.batchnorm_1(x)
        x = self.activation(x)
        layer_outputs = []
        for i in range(9):
            self.num_features *= 2 if (i==2 or i==8) else 1
            x = self.block(x, num_features, weight_decay, amplifying_ratio)
            layers_outputs.append(x)

        x = self.concat
'''   

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
    LeakyReLU
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


def se_fn(x, amplifying_ratio):
    num_features = x.shape[-1]
    x = GlobalAvgPool1D()(x)
    x = Reshape((1, num_features))(x)
    x = Dense(
        num_features * amplifying_ratio,
        kernel_initializer="glorot_uniform",
    )(x)
    #x = LeakyReLU(0.3)(x)
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
    x = Conv1D(num_features, kernel_size=3, padding='same', use_bias=True,
             kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
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
    #x = LeakyReLU(0.3)(x)
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
    #x = LeakyReLU(0.3)(x)
    x = Activation("relu")(x)
    x = MaxPool1D(pool_size=3)(x)
    return x


def resemul(
    block_type="se", amplifying_ratio=16, drop_rate=0.5, weight_decay=1e-4, num_classes=50,
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
    inp = Input(shape=(59049, 1))
    # Strided Conv
    num_features = 128
    x = Conv1D(
        num_features,
        kernel_size=3,
        strides=3,
        padding="valid",
        use_bias=True,
        kernel_regularizer=l2(weight_decay),
        kernel_initializer="he_uniform",
    )(inp)
    x = BatchNormalization()(x)
    #x = LeakyReLU(0.3)(x)
    x = Activation("relu")(x)

    # rese-block
    layer_outputs = []
    for i in range(9):
        num_features *= 2 if (i == 2 or i == 8) else 1
        x = block(x, num_features, weight_decay, amplifying_ratio)
        layer_outputs.append(x)

    x = Concatenate()([GlobalMaxPool1D()(output) for output in layer_outputs[-3:]])
    x = Dense(x.shape[-1], kernel_initializer="glorot_uniform")(x)
    x = BatchNormalization()(x)
    #out = LeakyReLU(0.3)(x)
    out = Activation("relu")(x)
    # if drop_rate > 0.0:
    #    x = Dropout(drop_rate)(x)
    # out = Dense(num_classes, activation="sigmoid", kernel_initializer="glorot_uniform")(
    #    x
    # )

    model = Model(inputs=inp, outputs=out)
    return model

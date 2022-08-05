from keras.applications import VGG19
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Dense,
    Flatten,
    Input,
    LeakyReLU,
)
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model


class Generator(Model):
    def __init__(self, shape, gf, n_residual_blocks) -> None:
        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding="same")(
                layer_input
            )
            d = Activation("relu")(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding="same")(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=(2, 3))(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding="same")(u)
            u = Activation("relu")(u)
            return u

        img_lr = Input(shape=shape)

        c1 = Conv2D(64, kernel_size=9, strides=1, padding="same")(img_lr)
        c1 = Activation("relu")(c1)

        r = residual_block(c1, gf)
        for _ in range(n_residual_blocks - 1):
            r = residual_block(r, gf)

        c2 = Conv2D(64, kernel_size=3, strides=1, padding="same")(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        gen_hr = Conv2D(
            shape[2],
            kernel_size=9,
            strides=1,
            padding="same",
            activation="tanh",
        )(u2)

        super().__init__(img_lr, gen_hr)


class Discriminator(Model):
    def __init__(self, shape, df):
        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding="same")(
                layer_input
            )
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=shape)

        d1 = d_block(d0, df, bn=False)
        d2 = d_block(d1, df, strides=2)
        d3 = d_block(d2, df * 2)
        d4 = d_block(d3, df * 2, strides=2)
        d5 = d_block(d4, df * 4)
        d6 = d_block(d5, df * 4, strides=2)
        d7 = d_block(d6, df * 8)
        d8 = d_block(d7, df * 8, strides=2)

        d9 = Dense(df * 16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation="sigmoid")(d10)

        super().__init__(d0, validity)


class VGG(Model):
    def __init__(self: "VGG", shape: tuple[int, int, int]) -> None:
        """
        Builds a pre-trained VGG19 model that outputs image features
        extracted at the third block of the model
        """
        img = Input(shape=shape, name="image_input")
        vgg = VGG19(
            weights="imagenet",
            include_top=False,
            input_shape=shape,
        )

        x = vgg(img)
        x = Flatten(name="flatten")(x)
        x = Dense(4096, activation="relu", name="fc1")(x)
        x = Dense(4096, activation="relu", name="fc2")(x)
        x = Dense(8, activation="softmax", name="predictions")(x)

        super().__init__(inputs=img, outputs=x)

import keras
from keras.layers import BatchNormalization, Dense, Dropout, LeakyReLU
from keras.models import Sequential


class Base(Sequential):
    def __init__(self: "Generator", layers: list, name: str) -> None:
        super().__init__(layers=layers, name=name)

        self.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        )


class Generator(Base):
    def __init__(self: "Generator", input_size=100, output_size=784) -> None:
        super().__init__(
            [
                Dense(units=256, input_dim=input_size),
                LeakyReLU(0.2),
                BatchNormalization(momentum=0.8),
                Dense(units=512),
                Dense(units=1024),
                LeakyReLU(0.2),
                Dense(units=output_size, activation="tanh"),
            ],
            "cloudput_generator",
        )


class Discriminator(Base):
    def __init__(self: "Discriminator", input_size=784) -> None:
        super().__init__(
            [
                Dense(units=1024, input_dim=input_size),
                LeakyReLU(0.2),
                Dropout(0.2),
                Dense(units=512),
                LeakyReLU(0.2),
                Dropout(0.3),
                Dense(units=256),
                LeakyReLU(0.2),
                Dropout(0.3),
                Dense(units=128),
                LeakyReLU(0.2),
                Dense(units=1, activation="sigmoid"),
            ],
            "cloudput_discriminator",
        )
        self.trainable = False

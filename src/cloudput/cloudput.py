from math import prod
from os import listdir, mkdir, path
from typing import Optional

import numpy as np
from keras.layers import Input
from keras.models import Model
from tqdm import tqdm

from src.cloudput.models import Discriminator, Generator
from src.utils import save_images


class Cloudput(Model):
    resolution: tuple[int, int]

    __input_size: int
    __shape: tuple[int, int, int]
    __generator: Generator
    __discriminator: Discriminator
    __weights_dir: str

    def __init__(
        self: "Cloudput",
        input_size: int = 10,
        output_resolution: tuple[int, int] = (28, 28),
        weights_dir: str = "./weights/cloudput",
        weights_epoch: Optional[int] = None,
    ) -> None:
        shape = (*output_resolution[:2], 3)
        output_size = prod(shape)
        generator = Generator(input_size=input_size, output_size=output_size)
        discriminator = Discriminator(input_size=output_size)
        inp = Input(shape=(input_size,))
        out = discriminator(generator(inp))

        super().__init__(inputs=inp, outputs=out)

        self.resolution = output_resolution
        self.__input_size = input_size
        self.__weights_dir = weights_dir
        self.__shape = shape

        self.__generator = generator
        self.__discriminator = discriminator

        self.compile(loss="binary_crossentropy", optimizer="Adam")

        if path.exists(weights_dir):
            self.load_weights(weights_epoch)

    def generate(self: "Cloudput", imgs: np.ndarray) -> np.ndarray:
        predictions = self.__generator.predict(imgs, verbose=0)
        return predictions.reshape((len(imgs), *self.__shape))

    def noise(self: "Cloudput", batch_size: int) -> np.ndarray:
        return np.random.normal(0, 1, (batch_size, self.__input_size))

    def load_weights(
        self: "Cloudput",
        weights_epoch: Optional[int] = None,
    ) -> None:
        weights_path = self.__get_weights_path(weights_epoch)
        if path.exists(weights_path):
            print(f"loading weights file...: {weights_path}")
            super().load_weights(filepath=weights_path)

    def save_weights(
        self: "Cloudput",
        epoch: int,
        overwrite: bool = True,
        save_format=None,
        options=None,
    ) -> None:
        if not path.isdir(self.__weights_dir):
            mkdir(self.__weights_dir)
        super().save_weights(
            filepath=self.__get_weights_path(epoch),
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )

    def train(
        self: "Cloudput",
        data: np.ndarray,
        epochs: int = 20,
        batch_size: int = 128,
        sample_interval: int = 50,
        images_dir: str = "./output/cloudput",
    ) -> None:
        for epoch in range(epochs):
            print(f"{epoch=}")

            for _ in tqdm(range(batch_size)):
                self.train_once(data, batch_size)

            if epoch % sample_interval == 0:
                imgs = self.generate(self.noise(batch_size))
                save_images(imgs, dir=images_dir, name=f"{epoch}")
                self.save_weights(epoch)

    def train_once(
        self: "Cloudput",
        data: np.ndarray,
        batch_size: int,
    ) -> None:
        high = data.shape[0]
        random_index = np.random.randint(low=0, high=high, size=batch_size)
        real = data[random_index].reshape(batch_size, prod(self.__shape))
        pred = self.__generator.predict(self.noise(batch_size), verbose=0)
        valid = np.ones(batch_size)
        fake = np.zeros(batch_size)
        self.__discriminator.trainable = True
        self.__discriminator.train_on_batch(
            np.concatenate([real, pred]),
            np.concatenate([valid, fake]),
        )
        self.__discriminator.trainable = False
        self.train_on_batch(self.noise(batch_size), valid)
        return

    def __get_weights_path(
        self: "Cloudput",
        epoch: Optional[int] = None,
    ) -> str:
        # format: dir/cloudput-WxH-epoch.h5
        size_str = "x".join(str(x) for x in self.resolution)
        prefix = f"cloudput-{size_str}-"
        fileext = ".h5"
        if epoch is None:
            epochs = sorted(
                f.removeprefix(prefix).removesuffix(fileext)
                for f in listdir(self.__weights_dir)
                if size_str in f
            )
            epoch = int(epochs[-1]) if len(epochs) != 0 else 0
        return f"{self.__weights_dir}/{prefix}{epoch}{fileext}"


if __name__ == "__main__":
    from .loader import Loader

    dir = path.relpath("/inputs/abstract-art")

    gan = Cloudput(input_size=10, output_resolution=(512, 512))
    loader = Loader(gan.resolution)
    training_data = loader.load_as_is(dir)
    gan.train(training_data, epochs=50, sample_interval=10)

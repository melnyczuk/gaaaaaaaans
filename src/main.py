from os import mkdir, path
from typing import Optional, Union

import numpy as np

from src.utils import save_images

from .cloudput import Cloudput
from .srgan import SRGAN, SrganData, SrganLoader


class GAN:
    resolution: SrganData[tuple[int, int]]

    __cloudput: Cloudput
    __srgan: SRGAN
    __weights_dir: str

    def __init__(
        self: "GAN",
        input_resolution: tuple[int, int],
        scale_factor: Union[int, tuple[int, int]],
        input_size: int = 10,
        weights_dir: str = "./weights",
        weights_epoch: Optional[int] = None,
    ) -> None:
        self.__cloudput = Cloudput(
            input_size=input_size,
            output_resolution=input_resolution,
            weights_dir=f"./{weights_dir}/cloudput",
            weights_epoch=weights_epoch,
        )
        self.__srgan = SRGAN(
            input_resolution=input_resolution,
            scale_factor=scale_factor,
            weights_dir=f"./{weights_dir}/srgan",
            weights_epoch=weights_epoch,
        )
        self.resolution = self.__srgan.resolution

        if path.exists(weights_dir):
            self.__weights_dir = weights_dir
            self.load_weights(weights_epoch)

    def generate(self: "GAN", imgs: np.ndarray) -> np.ndarray:
        return self.__srgan.generate(self.__cloudput.generate(imgs))

    def load_weights(self: "GAN", weights_epoch: Optional[int] = None) -> None:
        self.__cloudput.load_weights(weights_epoch)
        self.__srgan.load_weights(weights_epoch)

    def save_weights(
        self: "GAN",
        epoch: int,
        overwrite: bool = True,
        save_format=None,
        options=None,
    ) -> None:
        if not path.isdir(self.__weights_dir):
            mkdir(self.__weights_dir)
        self.__cloudput.save_weights(
            epoch,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
        self.__srgan.save_weights(
            epoch,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )

    def train(
        self: "GAN",
        data: SrganData[np.ndarray],
        epochs: int = 20,
        sample_interval: int = 50,
        images_dir: str = "./output",
    ) -> None:
        batch_size = len(data.hi)

        for epoch in range(epochs):
            print(f"{epoch=}")

            self.__cloudput.train_once(data.lo, batch_size)
            self.__srgan.train_once(data)

            if epoch % sample_interval == 0:
                noises = self.__cloudput.noise(batch_size)
                imgs = self.generate(noises)
                save_images(imgs, dir=images_dir, name=f"{epoch}")
                self.save_weights(epoch)


if __name__ == "__main__":

    dir = path.relpath(
        "/Users/how/Downloads/b1e027cf-291f-4dbe-b81a-503007cba650"
    )

    gan = GAN(input_resolution=(32, 32), scale_factor=(2, 3))
    loader = SrganLoader(gan.resolution)
    training_data = loader.load(dir)
    gan.train(training_data, epochs=1, sample_interval=1)

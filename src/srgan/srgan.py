from os import listdir, mkdir, path
from typing import Iterable, Optional, Union

import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from src.srgan.loader import SrganData
from src.srgan.models import VGG, Discriminator, Generator
from src.utils import save_images


class SRGAN(Model):
    resolution: SrganData[tuple[int, int]]

    __input_shape: tuple[int, int, int]
    __output_shape: tuple[int, int, int]

    __generator: Generator
    __discriminator: Discriminator
    __vgg: VGG
    __patch: tuple[int, int, int]
    __weights_dir: str

    def __init__(
        self: "SRGAN",
        input_resolution: tuple[int, int],
        scale_factor: Union[int, tuple[int, int]],
        weights_dir: str = "./weights/srgan",
        weights_epoch: Optional[int] = None,
    ) -> None:
        factor = (
            (scale_factor, scale_factor)
            if isinstance(scale_factor, int)
            else scale_factor
        )

        output_resolution = (
            input_resolution[0] * factor[0] ** 2,
            input_resolution[1] * factor[1] ** 2,
        )

        input_shape = (*input_resolution, 3)
        output_shape = (*output_resolution, 3)

        n_residual_blocks = 16
        optimizer = Adam(0.0002, 0.5)

        gf = 64
        df = 64

        vgg = VGG(output_shape)
        vgg.trainable = False
        vgg.compile(
            loss="mse",
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        discriminator = Discriminator(output_shape, df)
        discriminator.compile(
            loss="mse",
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        generator = Generator(input_shape, gf, n_residual_blocks)

        img_hr = Input(shape=(output_shape))
        img_lr = Input(shape=(input_shape))

        fake_hr = generator(img_lr)

        fake_features = vgg(fake_hr)

        discriminator.trainable = False

        validity = discriminator(fake_hr)

        super().__init__([img_lr, img_hr], [validity, fake_features])
        self.compile(
            loss=["binary_crossentropy", "mse"],
            loss_weights=[1e-3, 1],
            optimizer=optimizer,
        )

        self.resolution = SrganData(lo=input_resolution, hi=output_resolution)
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__patch = (
            int(output_resolution[0] / 2**4),
            int(output_resolution[1] / 2**4),
            1,
        )
        self.__weights_dir = weights_dir

        self.__generator = generator
        self.__discriminator = discriminator
        self.__vgg = vgg

        if path.exists(weights_dir):
            self.load_weights(weights_epoch)

    def generate(self: "SRGAN", imgs: Iterable[np.ndarray]) -> np.ndarray:
        return self.__generator.predict(imgs, verbose=0)

    def load_weights(
        self: "SRGAN",
        weights_epoch: Optional[int] = None,
    ) -> None:
        weights_path = self.__get_weights_path(weights_epoch)
        if path.exists(weights_path):
            print(f"loading weights file...: {weights_path}")
            super().load_weights(filepath=weights_path)

    def save_weights(
        self: "SRGAN",
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
        self: "SRGAN",
        data: SrganData[np.ndarray],
        epochs: int = 20,
        sample_interval: int = 50,
        images_dir: str = "./output/srgan",
    ) -> None:
        for epoch in range(epochs):
            print(f"{epoch=}")

            self.train_once(data)

            if epoch % sample_interval == 0:
                high = data.lo.shape[0]
                indices = np.random.randint(low=0, high=high, size=10)
                imgs = self.generate(data.lo[indices])
                save_images(imgs, dir=images_dir, name=f"{epoch}")
                self.save_weights(epoch)

    def train_once(self, data: SrganData[np.ndarray]) -> None:
        if (
            data.hi[0].shape != self.__output_shape
            or data.lo[0].shape != self.__input_shape
        ):
            raise Exception("SRGAN and training data shapes do not match")

        batch_size = len(data.hi)
        fake_hr = self.__generator.predict(data.lo, verbose=0)
        valid = np.ones((batch_size,) + self.__patch)
        fake = np.zeros((batch_size,) + self.__patch)
        self.__discriminator.trainable = True
        self.__discriminator.train_on_batch(
            np.concatenate([data.hi, fake_hr]),
            np.concatenate([valid, fake]),
        )
        self.__discriminator.trainable = False
        features = self.__vgg.predict(data.hi, verbose=0)
        self.train_on_batch([data.lo, data.hi], [valid, features])
        return

    def __get_weights_path(
        self: "SRGAN",
        epoch: Optional[int] = None,
    ) -> str:
        # format: dir/cloudput-WxH_WxH-epoch.h5
        size_str_hi = "x".join(str(x) for x in self.resolution.hi)
        size_str_lo = "x".join(str(x) for x in self.resolution.lo)
        size_str = f"{size_str_hi}_{size_str_lo}"
        prefix = f"srgan-{size_str}-"
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
    from .loader import Loader, SrganData

    dir = path.relpath(
        "/Users/how/Downloads/b1e027cf-291f-4dbe-b81a-503007cba650"
    )
    gan = SRGAN(input_resolution=(32, 32), scale_factor=(2, 3))
    loader = Loader(gan.resolution)
    training_data = loader.load(dir)
    gan.train(training_data, epochs=1, sample_interval=1)

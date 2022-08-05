from dataclasses import dataclass
from math import prod
from os import listdir, mkdir, path

import numpy as np
from PIL import Image


@dataclass
class Loader:
    __shape: tuple[int, int]

    def __init__(self: "Loader", shape: tuple[int, int]) -> None:
        self.__shape = (shape[1], shape[0])

    def prepare_and_load(self: "Loader", dir: str) -> np.ndarray:
        self.__prepare(dir)
        sized_dir = self.__get_sized_dir(dir)
        return self.load_as_is(sized_dir)

    def load_as_is(self: "Loader", dir: str) -> np.ndarray:
        return self.__normalise_data(
            np.array(
                [
                    np.array(
                        Image.open(path.join(dir, file_name)).convert("RGB")
                    )
                    for file_name in listdir(dir)
                ],
                dtype=np.uint8,
            )
        )

    def __get_sized_dir(self: "Loader", dir: str) -> str:
        return f"{dir}/{self.__shape[0]}:{self.__shape[1]}"

    def __normalise_data(self: "Loader", data: np.ndarray) -> np.ndarray:
        batch_size, *shape = data.shape
        normalised = data / 127.5 - 1.0
        return normalised.reshape(batch_size, prod(shape))

    def __prepare(self: "Loader", dir: str) -> None:
        for file_name in listdir(dir):
            if (ext := path.splitext(file_name)[1]) and ext is not None:
                in_path = path.join(dir, file_name)

                if not path.isdir(out_dir := self.__get_sized_dir(dir)):
                    mkdir(out_dir)

                if not path.exists(out_path := path.join(out_dir, file_name)):
                    (
                        Image.open(in_path)
                        .convert("RGB")
                        .resize(self.__shape)
                        .save(out_path)
                    )

from dataclasses import dataclass
from os import listdir, mkdir, path
from typing import Generic, TypeVar

import numpy as np
from PIL import Image

T = TypeVar("T")


@dataclass
class SrganData(Generic[T]):
    hi: T
    lo: T


class Loader:
    __shape: SrganData[tuple[int, int]]

    def __init__(self: "Loader", shape: SrganData[tuple[int, int]]) -> None:
        self.__shape = SrganData[tuple[int, int]](
            hi=(shape.hi[1], shape.hi[0]),
            lo=(shape.lo[1], shape.lo[0]),
        )

    def load(self: "Loader", dir: str) -> SrganData[np.ndarray]:
        self.__prepare(dir)
        sized_dirs = self.__get_sized_dir(dir)
        return self.__normalise_data(
            SrganData[np.ndarray](
                hi=np.array(
                    [
                        np.array(
                            Image.open(
                                path.join(sized_dirs.hi, file_name)
                            ).convert("RGB")
                        )
                        for file_name in listdir(sized_dirs.hi)
                    ],
                    dtype=np.uint8,
                ),
                lo=np.array(
                    [
                        np.array(
                            Image.open(
                                path.join(sized_dirs.lo, file_name)
                            ).convert("RGB")
                        )
                        for file_name in listdir(sized_dirs.lo)
                    ],
                    dtype=np.uint8,
                ),
            )
        )

    def __get_sized_dir(self: "Loader", dir: str) -> SrganData[str]:
        return SrganData[str](
            hi=f"{dir}/{self.__shape.hi[0]}:{self.__shape.hi[1]}",
            lo=f"{dir}/{self.__shape.lo[0]}:{self.__shape.lo[1]}",
        )

    def __normalise_data(
        self: "Loader",
        data: SrganData[np.ndarray],
    ) -> SrganData[np.ndarray]:
        def normalise(x: np.ndarray) -> np.ndarray:
            return x / 127.5 - 1.0

        return SrganData[np.ndarray](
            hi=normalise(data.hi),
            lo=normalise(data.lo),
        )

    def __prepare(self: "Loader", dir: str) -> None:
        for file_name in listdir(dir):
            if (ext := path.splitext(file_name)[1]) and ext is not None:
                in_path = path.join(dir, file_name)
                out_dirs = self.__get_sized_dir(dir)

                if not path.isdir(out_dirs.hi):
                    mkdir(out_dirs.hi)
                if not path.isdir(out_dirs.lo):
                    mkdir(out_dirs.lo)

                if not path.exists(
                    out_path_hi := path.join(out_dirs.hi, file_name)
                ):
                    (
                        Image.open(in_path)
                        .convert("RGB")
                        .resize(self.__shape.hi)
                        .save(out_path_hi)
                    )

                if not path.exists(
                    out_path_lo := path.join(out_dirs.lo, file_name)
                ):
                    (
                        Image.open(in_path)
                        .convert("RGB")
                        .resize(self.__shape.lo)
                        .save(out_path_lo)
                    )

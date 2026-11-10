import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray, ArrayLike
from typing import Any, Union


@dataclass
class AudioSignal:  # to avoid name conflict with scipy signal module
    data: NDArray[np.float64]
    srate: int

    @property
    def duration(self) -> float:
        return len(self.data) / self.srate

    @property
    def numchannels(self) -> int:
        return self.data.shape[0] if self.data.ndim > 1 else 1

    @property
    def quantization(self) -> type[Any]:
        return type(self.data[0])

    def _normalize_data(self, data: ArrayLike) -> NDArray[np.float64]:
        assert isinstance(data, (np.ndarray, list, tuple)), "'data' must be array-like"
        assert len(data) > 0, "'data' must not be empty"

        #
        max_abs = np.max(np.abs(data))
        if max_abs > 1.0:
            data = data / max_abs
        return np.asarray(data, dtype=np.float64)

    def _format_srate(self, sr: Union[int, float]) -> int:
        try:
            sr = int(sr)
        except Exception as e:
            raise TypeError(f"Sampling rate not transformable to integer: {e}")
        return sr

    def _validate(self) -> None:
        if not isinstance(self.srate, int) or self.srate <= 0:
            raise ValueError("Sampling rate must be a positive integer.")
        if self.data.max() > 1.0 or self.data.min() < -1.0:
            raise ValueError("Data values must be in the range [-1.0, 1.0].")

    def __init__(self, data: ArrayLike, srate: int) -> None:
        self.data = self._normalize_data(data)
        self.srate = self._format_srate(srate)
        self._validate()

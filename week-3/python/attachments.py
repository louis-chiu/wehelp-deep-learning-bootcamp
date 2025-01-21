from dataclasses import dataclass
from abc import ABC, abstractmethod

"""
TODO: 
 * 思考 Netwok class 所需的 arguments
 * 是否需要每次都自定義 hidden layer 的參數及權重？
"""


class Layer(ABC):
    @abstractmethod
    def forward(self, inputs) -> None:
        raise NotImplementedError("Each layer must implement forward pass")


class FullyConttectedLayer(Layer):
    def __init__(
        self,
        *,
        input_size: int | None,
        units: int | None,
        weight: list[list[float]] | None = None,
        bias: list[float] | None = None,
    ):
        self._input_size = input_size
        self._units = units

    @property
    def input_size(self) -> int | None:
        return self._input_size

    @property
    def units(self) -> int | None:
        return self._units

    def forward(self, inputs):
        pass

    def _validate_initial_params(self):
        pass

    def _validate_weight(self):
        pass

    def _validate_bias(self):
        pass


@dataclass
class NetworkConfig:
    input_size: int
    output_size: int
    layers: list[Layer]


class Network:
    def __init__(self, config: NetworkConfig):
        self.layers: list[Layer] = config.layers

    def forward(self, input: list[float]) -> list[float]:
        for layer in self.layers:
            layer.forward()


# @ operator: matrix multiplication
# https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc

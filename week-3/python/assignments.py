from abc import ABC, abstractmethod
from typing import Self
from random import random


class Layer(ABC):
    @abstractmethod
    def forward(self, inputs) -> None:
        raise NotImplementedError("Each layer must implement forward pass")


class FullyConnectedLayer(Layer):
    def __init__(
        self,
        *,
        input_size: int | None = None,
        units: int | None = None,
        weight: list[list[float]] | None = None,
        bias: list[float] | None = None,
    ):
        self._input_size = input_size
        self._units = units
        self._weight = weight
        self._bias = bias

        self._initialize_weight_and_bias()

    @property
    def input_size(self) -> int | None:
        return self._input_size

    @property
    def units(self) -> int | None:
        return self._units

    @property
    def weight(self) -> list[list[float]] | None:
        return self._weight

    @property
    def bias(self) -> list[float] | None:
        return self._bias

    def forward(self, input_: list[float]) -> list[float]:
        if len(input_) != self.input_size:
            raise ValueError(
                f"Invalid argument input: expected {self.input_size} rows, but got {len(input_)}"
            )
        return [
            sum([w * x for w, x in zip(weight_row, input_)]) + b
            for weight_row, b in zip(self.weight, self.bias)
        ]

    def _initialize_weight_and_bias(self):
        if self.weight is not None:
            if self.input_size is not None and len(self.weight[0]) != self.input_size:
                raise ValueError(
                    f"Invalid weight: expected {self.input_size} rows, but got {len(self.weight[0])}"
                )
            self._input_size = len(self.weight[0])

            if self.units is not None and len(self.weight) != self.units:
                raise ValueError(
                    f"Invalid argument weight: expected {self.units} columns, but got {len(self.weight)}"
                )
            self._units = len(self.weight)
        else:
            if self.input_size is None or self.units is None:
                raise ValueError(
                    "Invalid arguments: at least pass 'input_size' and 'units' to initialize FullyConnectedLayer object."
                )
            self._weight = [
                [random() for _ in range(self.input_size)] for _ in range(self.units)
            ]

        if self.bias is not None:
            if self.units is not None and len(self.bias) != self.units:
                raise ValueError(
                    f"Invalid argument bias: expected {self.units} elements, but got {len(self.bias)}"
                )
        else:
            if self.units is None:
                raise ValueError(
                    "Invalid arguments: at least pass 'input_size' and 'units' to initialize FullyConnectedLayer object."
                )
            self._bias = [random() for _ in range(self.units)]


class Network:
    def __init__(self, layers: list[Layer]):
        self._layers = layers

    @property
    def layers(self) -> list[Layer]:
        return self._layers

    def _forward_layer(self, layer_input, index, layer):
        return layer.forward(layer_input)

    def forward(self, input_: list[float]) -> list[float]:
        layer_input = input_
        for index, layer in enumerate(self.layers):
            try:
                layer_input = layer.forward(layer_input)
            except ValueError as e:
                raise ValueError(f"{e} (Error happens at index {index} of layers)")

        return layer_input

    class Builder:
        def __init__(self):
            self.layers = []

        def add_fully_connected_layer(
            self,
            *,
            input_size: int | None = None,
            units: int | None = None,
            weight: list[list[float]] | None = None,
            bias: list[float] | None = None,
        ) -> Self:
            self.layers.append(
                FullyConnectedLayer(
                    input_size=input_size, units=units, weight=weight, bias=bias
                )
            )

            return self

        def build(self):
            return Network(self.layers)

    @staticmethod
    def builder() -> Builder:
        return Network.Builder()


def neural_network_1():
    network = Network(
        [
            FullyConnectedLayer(
                weight=[
                    [0.5, 0.2],
                    [0.6, -0.6],
                ],
                bias=[0.3, 0.25],
            ),
            FullyConnectedLayer(weight=[[0.8, 0.4]], bias=[-0.5]),
        ]
    )

    input_1 = [1.5, 0.5]
    input_2 = [0, 1]
    print("=================== Model 1 ===================")
    print(f"Input: {input_1}")
    print(f"Output: {network.forward(input_1)}")
    print()
    print(f"Input: {input_2}")
    print(f"Output: {network.forward(input_2)}")


def neural_network_2():
    network = (
        Network.builder()
        .add_fully_connected_layer(weight=[[0.5, 1.5], [0.6, -0.8]], bias=[0.3, 1.25])
        .add_fully_connected_layer(weight=[[0.6, -0.8]], bias=[0.3])
        .add_fully_connected_layer(weight=[[0.5], [-0.4]], bias=[0.2, 0.5])
        .build()
    )
    input_1 = [0.75, 1.25]
    input_2 = [-1, 0.5]
    print("=================== Model 2 ===================")
    print(f"Input: {input_1}")
    print(f"Output: {network.forward(input_1)}")
    print()
    print(f"Input: {input_2}")
    print(f"Output: {network.forward(input_2)}")


if __name__ == "__main__":
    neural_network_1()
    neural_network_2()

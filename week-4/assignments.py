from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self
from random import random
from enum import StrEnum
import math


class Layer(ABC):
    @abstractmethod
    def forward(self, inputs) -> None:
        raise NotImplementedError("Each layer must implement forward pass")


class Activation(StrEnum):
    RELU = "relu"
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


class ActivationUtils:
    @staticmethod
    def linear(x: float) -> float:
        return x

    @staticmethod
    def relu(x: float) -> float:
        return x if x > 0 else 0

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def softmax(input_: list[float]) -> list[float]:
        max_val = max(input_)
        exp_values = [math.exp(x - max_val) for x in input_]
        sum_exp = sum(exp_values)
        return [x / sum_exp for x in exp_values]

    @staticmethod
    def get_activation(
        type: Activation,
    ) -> Callable[[float], float] | Callable[[list[float]], list[float]]:
        match type:
            case Activation.RELU:
                return ActivationUtils.relu
            case Activation.SIGMOID:
                return ActivationUtils.sigmoid
            case Activation.LINEAR:
                return ActivationUtils.linear
            case Activation.SOFTMAX:
                return ActivationUtils.softmax
            case _:
                raise ValueError(f"Invalid argument input: {type}")


class LossFunctionUtils:
    @staticmethod
    def mean_square_error(output: list[float], expects: list[float]) -> float:
        if len(output) != len(expects):
            raise ValueError("Output and expected lists must have the same length.")

        n = len(output)
        return (1 / n) * sum(math.pow(e - o, 2) for o, e in zip(output, expects))

    @staticmethod
    def binary_cross_entropy(output: list[float], expects: list[int]) -> float:
        if len(output) != len(expects):
            raise ValueError("Output and expected lists must have the same length.")

        return -sum(
            [
                (e * math.log(o) + (1 - e) * math.log(1 - o))
                for o, e in zip(output, expects)
            ]
        )

    @staticmethod
    def categorical_cross_entropy(
        output: list[float], expects: list[float]
    ) -> list[float]:
        if len(output) != len(expects):
            raise ValueError("Output and expected lists must have the same length.")

        return -sum([e * math.log(o) for o, e in zip(output, expects)])


class FullyConnectedLayer(Layer):
    def __init__(
        self,
        *,
        input_size: int | None = None,
        units: int | None = None,
        weight: list[list[float]] | None = None,
        bias: list[float] | None = None,
        activation: Activation = Activation.LINEAR,
    ):
        self._input_size = input_size
        self._units = units
        self._weight = weight
        self._bias = bias
        self._activation = activation

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

        if self._activation == Activation.SOFTMAX:
            return ActivationUtils.softmax(
                [
                    sum([w * x for w, x in zip(weight_row, input_)]) + b
                    for weight_row, b in zip(self.weight, self.bias)
                ]
            )
        activation: Callable[[float], float] = ActivationUtils.get_activation(
            self._activation
        )
        return [
            activation(sum([w * x for w, x in zip(weight_row, input_)]) + b)
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
            activation: Activation = Activation.LINEAR,
        ) -> Self:
            self.layers.append(
                FullyConnectedLayer(
                    input_size=input_size,
                    units=units,
                    weight=weight,
                    bias=bias,
                    activation=activation,
                )
            )

            return self

        def build(self):
            return Network(self.layers)

    @staticmethod
    def builder() -> Builder:
        return Network.Builder()


def regression_task():
    nn = (
        Network.builder()
        .add_fully_connected_layer(
            weight=[[0.5, 0.2], [0.6, -0.6]], bias=[0.3, 0.25], activation="relu"
        )
        .add_fully_connected_layer(weight=[[0.8, -0.5], [0.4, 0.5]], bias=[0.6, -0.25])
        .build()
    )
    print("=================== Model 1 ===================")

    print(
        f"Total Loss: {LossFunctionUtils.mean_square_error(nn.forward([1.5, 0.5]), [0.8, 1])}"
    )

    print(
        f"Total Loss: {LossFunctionUtils.mean_square_error(nn.forward([0, 1]), [0.5, 0.5])}"
    )


def binary_classification_task():
    nn = (
        Network.builder()
        .add_fully_connected_layer(
            weight=[[0.5, 0.2], [0.6, -0.6]], bias=[0.3, 0.25], activation="relu"
        )
        .add_fully_connected_layer(
            weight=[[0.8, 0.4]], bias=[-0.5], activation="sigmoid"
        )
        .build()
    )
    print("=================== Model 2 ===================")

    print(
        f"Total Loss: {LossFunctionUtils.binary_cross_entropy(nn.forward([0.75, 1.25]), [1])}"
    )

    print(
        f"Total Loss: {LossFunctionUtils.binary_cross_entropy(nn.forward([-1, 0.5]), [0])}"
    )


def multiple_label_classification_task():
    nn = (
        Network.builder()
        .add_fully_connected_layer(
            weight=[[0.5, 0.2], [0.6, -0.6]], bias=[0.3, 0.25], activation="relu"
        )
        .add_fully_connected_layer(
            weight=[[0.8, -0.4], [0.5, 0.4], [0.3, 0.75]],
            bias=[0.6, 0.5, -0.5],
            activation="sigmoid",
        )
        .build()
    )
    print("=================== Model 3 ===================")

    print(
        f"Total Loss: {LossFunctionUtils.binary_cross_entropy(nn.forward([1.5, 0.5]), [1, 0, 1])}"
    )

    print(
        f"Total Loss: {LossFunctionUtils.binary_cross_entropy(nn.forward([0, 1]), [1, 1, 0])}"
    )


def multiple_class_classification_task():
    nn = (
        Network.builder()
        .add_fully_connected_layer(
            weight=[[0.5, 0.2], [0.6, -0.6]], bias=[0.3, 0.25], activation="relu"
        )
        .add_fully_connected_layer(
            weight=[[0.8, -0.4], [0.5, 0.4], [0.3, 0.75]],
            bias=[0.6, 0.5, -0.5],
            activation="softmax",
        )
        .build()
    )
    print("=================== Model 4 ===================")

    print(
        f"Total Loss: {LossFunctionUtils.categorical_cross_entropy(nn.forward([1.5, 0.5]), [1, 0, 0])}"
    )

    print(
        f"Total Loss: {LossFunctionUtils.categorical_cross_entropy(nn.forward([0, 1]), [0, 0, 1])}"
    )


def main():
    regression_task()
    binary_classification_task()
    multiple_label_classification_task()
    multiple_class_classification_task()


if __name__ == "__main__":
    main()

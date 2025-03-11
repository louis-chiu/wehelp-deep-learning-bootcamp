from abc import ABC, abstractmethod
from typing import Self
import random
import math
from activation import Activation, ActivationBase, get_activation


class Layer(ABC):
    @abstractmethod
    def forward(self, inputs: list[float]) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def backward(self, losses: list[float]) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self, learning_rate: float) -> None:
        raise NotImplementedError


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

        self._input = None
        self._output = None
        self._weight_gradients = None
        self._bias_gradients = None

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

    @property
    def gradients(self) -> list[list[float]] | None:
        return self._gradients

    def forward(self, input_: list[float]) -> list[float]:
        self._input = input_

        if len(input_) != self.input_size:
            raise ValueError(
                f"Invalid argument input: expected {self.input_size} rows, but got {len(input_)}"
            )

        activation: ActivationBase = get_activation(self._activation)
        if self._activation == Activation.SOFTMAX:
            self._output = activation.apply(
                [
                    sum([w * x for w, x in zip(weight_row, input_)]) + b
                    for weight_row, b in zip(self.weight, self.bias)
                ]
            )
        else:
            self._output = [
                activation.apply(sum([w * x for w, x in zip(weight_row, input_)]) + b)
                for weight_row, b in zip(self.weight, self.bias)
            ]
        return self._output

    def backward(self, losses: list[float]) -> list[float]:
        if self._output is None:
            raise ValueError("forward has not been performed yet")

        activation: ActivationBase = get_activation(self._activation)

        if self._activation == Activation.SOFTMAX:
            raise NotImplementedError

        # activation_gradients: ∂y/∂z
        activation_gradients = [activation.derivative(o) for o in self._output]

        # losses: ∂C/∂y
        # delta: ∂C/∂z = ∂C/∂y * ∂y/∂z
        delta = [
            loss * gradient for loss, gradient in zip(losses, activation_gradients)
        ]

        # self._input: ∂z/∂w
        # weight_gradients: ∂C/∂w = ∂C/∂z * ∂z/∂w
        weight_gradients = [[d * x for x in self._input] for d in delta]

        # bias_gradients: ∂C/∂z
        bias_gradients = delta

        # prev_layer_gradient: ∂C/∂x = ∂C/∂z * ∂z/∂a * ∂a/x
        prev_layer_gradient = [
            sum(w * delta_val for w, delta_val in zip(weight_col, delta))
            for weight_col in zip(*self.weight)
        ]

        self._bias_gradients = bias_gradients
        self._weight_gradients = weight_gradients
        return prev_layer_gradient

    def zero_grad(self, learning_rate: float):
        self._weight = [
            [w - (learning_rate * g) for w, g in zip(weight_row, gradient_row)]
            for weight_row, gradient_row in zip(self.weight, self._weight_gradients)
        ]
        self._bias = [
            b - (learning_rate * g) for b, g in zip(self.bias, self._bias_gradients)
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
            if self._activation != Activation.RELU:
                # Xavier
                limit = math.sqrt(6 / (self.input_size + self.units))
            else:
                # He
                limit = math.sqrt(6 / self.input_size)

            self._weight = [
                [random.uniform(-limit, limit) for _ in range(self.input_size)]
                for _ in range(self.units)
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
            self._bias = [0 for _ in range(self.units)]


class Network:
    def __init__(self, layers: list[Layer]):
        self._layers = layers

    @property
    def layers(self) -> list[Layer]:
        return self._layers

    def _forward_layer(self, layer_input, layer):
        return layer.forward(layer_input)

    def forward(self, input_: list[float]) -> list[float]:
        layer_input = input_
        for index, layer in enumerate(self.layers):
            try:
                layer_input = layer.forward(layer_input)
            except Exception as e:
                raise ValueError(f"{e} (Error happens at index {index} of layers)")

        return layer_input

    def backward(self, losses: list[float]):
        current_gradient = losses
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient)

    def zero_grad(self, learning_rate: float):
        for index, layer in enumerate(self.layers):
            try:
                layer.zero_grad(learning_rate)
            except ValueError as e:
                raise ValueError(f"{e} (Error happens at index {index} of layers)")

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

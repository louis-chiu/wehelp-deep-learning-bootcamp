from abc import ABC, abstractmethod
from enum import StrEnum
import math


class ActivationBase(ABC):
    @abstractmethod
    def apply(self, x: float | list[float]) -> float | list[float]:
        pass

    def derivative(self, x: float | list[float]) -> float | list[float]:
        pass


class Linear(ActivationBase):
    def apply(self, x: float) -> float:
        return x

    def derivative(self, x: float) -> float:
        return 1


class ReLU(ActivationBase):
    def apply(self, x: float) -> float:
        return max(x, 0)

    def derivative(self, x: float) -> float:
        return 1 if x > 0 else 0


class Sigmoid(ActivationBase):
    def apply(self, x: float) -> float:
        return 1 / (1 + safe_exp(-x))

    def derivative(self, x: float) -> float:
        return self.apply(x) * (1 - self.apply(x))


class Softmax(ActivationBase):
    def apply(self, input_: list[float]) -> list[float]:
        max_val = max(input_)
        exp_values = [safe_exp(x - max_val) for x in input_]
        sum_exp = sum(exp_values)
        return [x / sum_exp for x in exp_values]

    def derivative(self, x):
        raise NotImplementedError


class Activation(StrEnum):
    RELU = "relu"
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


_activation_instances = {}


def get_activation(
    type: Activation,
) -> ActivationBase:
    if type not in _activation_instances:
        match type:
            case Activation.RELU:
                _activation_instances[type] = ReLU()
            case Activation.SIGMOID:
                _activation_instances[type] = Sigmoid()
            case Activation.LINEAR:
                _activation_instances[type] = Linear()
            case Activation.SOFTMAX:
                _activation_instances[type] = Softmax()
            case _:
                raise ValueError(f"Invalid activation type: {type}")

    return _activation_instances[type]


def safe_exp(x: float):
    if -745 > x:
        return 0
    elif x > 709:
        return math.exp(709)

    return math.exp(x)

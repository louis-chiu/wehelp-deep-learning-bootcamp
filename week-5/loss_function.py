import math
from abc import ABC, abstractmethod


class LossFunction(ABC):
    @abstractmethod
    def get_total_loss(self, output: list[float], expects: list[float]) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_output_losses(
        self, output: list[float], expects: list[float]
    ) -> list[float]:
        raise NotImplementedError


class MeanSqureError(LossFunction):
    def get_total_loss(self, output, expects):
        if len(output) != len(expects):
            raise ValueError("Output and expected lists must have the same length.")

        n = len(output)
        return (1 / n) * sum(math.pow(e - o, 2) for o, e in zip(output, expects))

    def get_output_losses(self, output, expects):
        if len(output) != len(expects):
            raise ValueError("Output and expected lists must have the same length.")

        n = len(output)
        return [2 / n * (o - e) for o, e in zip(output, expects)]


class BinaryCrossEntropy(LossFunction):
    def get_total_loss(self, output, expects):
        if len(output) != len(expects):
            raise ValueError("Output and expected lists must have the same length.")
        
        return -sum(
            [
                (e * math.log(o) + (1 - e) * math.log(1 - o))
                for o, e in zip(output, expects)
            ]
        )

    def get_output_losses(self, output, expects):
        if len(output) != len(expects):
            raise ValueError("Output and expected lists must have the same length.")

        return [-(e / o) + ((1 - e) / (1 - o)) for o, e in zip(output, expects)]


class CategoricalCrossEntropy(LossFunction):
    def get_total_loss(self, output, expects):
        if len(output) != len(expects):
            raise ValueError("Output and expected lists must have the same length.")

        return -sum([e * math.log(o) for o, e in zip(output, expects)])

    def get_output_losses(self, output, expects):
        return super().get_output_losses(expects)


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

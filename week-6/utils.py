from statistics import pstdev, mean


class MathUtils:
    @staticmethod
    def mean(numbers: list[int | float]) -> int | float:
        if len(numbers) == 0:
            raise ValueError("List of numbers may not be empty.")

        return sum(numbers) / len(numbers)

    @staticmethod
    def stdev(numbers: list[int | float]) -> int | float:
        if len(numbers) == 0:
            raise ValueError("List of number may not be empty.")

        mu = MathUtils.mean(numbers)
        return (sum([(number - mu) ** 2 for number in numbers]) / len(numbers)) ** 0.5

    @staticmethod
    def zscore(number: int | float, mu: int | float, sigma: int | float) -> int | float:
        if sigma == 0:
            raise ValueError("Sigma must not be 0.")

        return (number - mu) / sigma

    @staticmethod
    def inverse_zscore(
        number: int | float, mu: int | float, sigma: int | float
    ) -> int | float:
        if sigma == 0:
            raise ValueError("Sigma must not be 0.")

        return number * sigma + mu

    @staticmethod
    def inverse_zscore_all(
        z_scores: list[int | float], mu: int | float, sigma: int | float
    ) -> list[int | float]:
        if sigma == 0:
            raise ValueError("Sigma must not be 0.")

        return [MathUtils.inverse_zscore(z_score, mu, sigma) for z_score in z_scores]

    @staticmethod
    def get_mean_of(data: list[tuple[float, ...] | float], index: int = None) -> float:
        if index is None:
            return mean(data)

        return mean([datum[index] for datum in data])

    @staticmethod
    def get_pstdev_of(
        data: list[tuple[float, ...] | float], index: int = None
    ) -> float:
        if index is None:
            return pstdev(data)

        return pstdev([datum[index] for datum in data])

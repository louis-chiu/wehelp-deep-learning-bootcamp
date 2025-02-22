import csv
from os import path, getcwd
from statistics import pstdev, mean
from network import Network
from loss_function import MeanSquareError
from math import sqrt
from utils import MathUtils


class Task1:
    @staticmethod
    def extract(
        file_path: str,
    ) -> tuple[list[int], list[tuple[float, float]]]:
        try:
            with open(file_path) as file:
                lines = csv.reader(file)
                next(lines)

                input_data = []
                expects_data = []
                for gender, height, weight in lines:
                    input_data.append((gender, float(height)))
                    expects_data.append(float(weight))

                return input_data, expects_data
        except Exception as e:
            print(f"An error occurred while processing the file {file_path}: {e}")
            raise e

    @staticmethod
    def transform(
        input_data: list[tuple[float, float]], expects_data: list[tuple[float]]
    ) -> tuple[list[tuple[float, float]], list[tuple[float]]]:
        weight_mean, height_mean = (
            Task1.get_mean_of(expects_data),
            Task1.get_mean_of(input_data, 1),
        )

        weight_stdev, height_stdev = (
            Task1.get_pstdev_of(expects_data),
            Task1.get_pstdev_of(input_data, 1),
        )

        return (
            [
                (
                    Task1.transform_gender(gender),
                    MathUtils.zscore(height, height_mean, height_stdev),
                )
                for gender, height in input_data
            ],
            [
                (MathUtils.zscore(weight, weight_mean, weight_stdev),)
                for weight in expects_data
            ],
        )

    @staticmethod
    def transform_gender(gender: str) -> int:
        match gender:
            case "Male":
                return 1
            case "Female":
                return 0
            case _:
                return -1

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

    @staticmethod
    def run():
        file_path = path.join(getcwd(), "week-6/dataset/gender-height-weight.csv")

        input_data, expects_data = Task1.extract(file_path)
        weight_mean, weight_stdev = (
            Task1.get_mean_of(expects_data),
            Task1.get_pstdev_of(expects_data),
        )
        input_data, expects_data = Task1.transform(input_data, expects_data)

        nn = (
            Network.builder()
            .add_fully_connected_layer(input_size=2, units=3, activation="relu")
            .add_fully_connected_layer(input_size=3, units=2, activation="relu")
            .add_fully_connected_layer(input_size=2, units=1)
            .build()
        )

        mse = MeanSquareError()
        learning_rate = 0.001
        loss_sum = 0
        for _ in range(5):
            for x, e in zip(input_data, expects_data):
                output = nn.forward(x)
                loss = mse.get_total_loss(
                    MathUtils.inverse_zscore_all(output, weight_mean, weight_stdev),
                    MathUtils.inverse_zscore_all(e, weight_mean, weight_stdev),
                )

                loss_sum += sqrt(loss)

                output_losses = mse.get_output_losses(output, e)
                nn.backward(output_losses)
                nn.zero_grad(learning_rate)

        avg_loss = loss_sum / len(input_data)
        print(f"Before Training Avg Loss in Weight: {avg_loss: .2f} pounds")

        loss_sum = 0
        for x, e in zip(input_data, expects_data):
            output = nn.forward(x)
            loss = mse.get_total_loss(
                MathUtils.inverse_zscore_all(output, weight_mean, weight_stdev),
                MathUtils.inverse_zscore_all(e, weight_mean, weight_stdev),
            )

            loss_sum += sqrt(loss)

        avg_loss = loss_sum / len(input_data)
        print(f"After Training Avg Loss in Weight: {avg_loss: .2f} pounds")


def main():
    Task1.run()


if __name__ == "__main__":
    main()

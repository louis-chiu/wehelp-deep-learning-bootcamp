import csv
from network import Network
import loss_function
from math import sqrt
from utils import MathUtils
from collections import OrderedDict
import torch


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
            MathUtils.get_mean_of(expects_data),
            MathUtils.get_mean_of(input_data, 1),
        )

        weight_stdev, height_stdev = (
            MathUtils.get_pstdev_of(expects_data),
            MathUtils.get_pstdev_of(input_data, 1),
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
    def run():
        print(f"{' Task 1: Weight Prediction ':=^75}")

        file_path = "dataset/gender-height-weight.csv"

        input_data, expects_data = Task1.extract(file_path)
        weight_mean, weight_stdev = (
            MathUtils.get_mean_of(expects_data),
            MathUtils.get_pstdev_of(expects_data),
        )
        input_data, expects_data = Task1.transform(input_data, expects_data)

        nn = (
            Network.builder()
            .add_fully_connected_layer(input_size=2, units=16, activation="relu")
            .add_fully_connected_layer(input_size=16, units=8, activation="relu")
            .add_fully_connected_layer(input_size=8, units=1)
            .build()
        )

        mse = loss_function.MeanSquareError()
        epoches = 20
        learning_rate = 0.001

        loss_sum = 0
        for epoch in range(epoches):
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
            if epoch == 0:
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
        print()


class Task2:
    @staticmethod
    def extract(
        file_path: str,
    ) -> dict[str, list]:
        try:
            with open(file_path) as file:
                lines = csv.reader(file)
                column_names = next(lines)

                data: dict[str, list] = OrderedDict([(key, []) for key in column_names])
                for line in lines:
                    for items, datum in zip(data.values(), line):
                        items.append(datum)
                return data

        except Exception as e:
            print(f"An error occurred while processing the file {file_path}: {e}")
            raise e

    @staticmethod
    def transform(
        data: dict[str, list],
    ) -> tuple[list[tuple[float, ...]], list[tuple[float]]]:
        (
            _,
            IS_SURVIVED,
            PCLASS,
            _,
            SEX,
            AGE,
            SIB_SP,
            PARCH,
            _,
            FARE,
            _,
            _,
        ) = data.keys()
        FAMILY_NUMS = "family_nums"
        data[PCLASS] = [float(datum) if datum else -1 for datum in data[PCLASS]]
        data[SEX] = [
            Task2.transform_gender(datum) if datum else -1 for datum in data[SEX]
        ]

        data[AGE] = [float(datum) if datum else None for datum in data[AGE]]
        data[FARE] = [float(datum) if datum else -1 for datum in data[FARE]]
        data[FAMILY_NUMS] = [
            float(sib_sp if sib_sp else 0 + parch if parch else 0)
            for sib_sp, parch in zip(data[SIB_SP], data[PARCH])
        ]
        age_mean, age_pstdev = (
            MathUtils.get_mean_of(filter(lambda x: x, data[AGE])),
            MathUtils.get_pstdev_of(filter(lambda x: x, data[AGE])),
        )
        family_nums_mean, family_nums_pstdev = (
            MathUtils.get_mean_of(data[FAMILY_NUMS]),
            MathUtils.get_pstdev_of(data[FAMILY_NUMS]),
        )
        fare_mean, fare_pstdev = (
            MathUtils.get_mean_of(data[FARE]),
            MathUtils.get_pstdev_of(data[FARE]),
        )

        input_data = [
            (
                1 if pclass == 1 else 0,
                1 if pclass == 2 else 0,
                1 if pclass == 3 else 0,
                sex,
                MathUtils.zscore(age if age else age_mean, age_mean, age_pstdev),
                MathUtils.zscore(family_nums, family_nums_mean, family_nums_pstdev),
                MathUtils.zscore(fare, fare_mean, fare_pstdev),
            )
            for pclass, sex, age, family_nums, fare in zip(
                data[PCLASS],
                data[SEX],
                data[AGE],
                data[FAMILY_NUMS],
                data[FARE],
            )
        ]
        expects_data = [(float(datum),) for datum in data[IS_SURVIVED]]

        return (input_data, expects_data)

    @staticmethod
    def transform_gender(gender: str) -> int:
        match gender:
            case "male":
                return 1
            case "female":
                return 0
            case _:
                return -1

    @staticmethod
    def run():
        print(f"{' Task 2: Titanic Survival Prediction ':=^75}")
        file_path = "dataset/titanic.csv"
        data = Task2.extract(file_path)
        input_data, expects_data = Task2.transform(data)

        nn = (
            Network.builder()
            .add_fully_connected_layer(input_size=len(input_data[0]), units=32)
            .add_fully_connected_layer(input_size=32, units=16, activation="relu")
            .add_fully_connected_layer(input_size=16, units=8, activation="relu")
            .add_fully_connected_layer(input_size=8, units=1, activation="sigmoid")
            .build()
        )

        bce = loss_function.BinaryCrossEntropy()
        learning_rate = 0.005
        epoches = 20

        correct_times = 0
        is_survived = 0
        for epoch in range(epoches):
            for x, e in zip(input_data, expects_data):
                output = nn.forward(x)
                is_survived = 1 if output[0] > 0.5 else 0

                if is_survived == e[0]:
                    correct_times += 1

                output_losses = bce.get_output_losses(output, e)
                nn.backward(output_losses)
                nn.zero_grad(learning_rate)
            if epoch == 0:
                accuracy = correct_times / len(input_data)
                print(f"Before Training Prediction Accuracy: {accuracy * 100: .2f} %")

        correct_times = 0
        for x, e in zip(input_data, expects_data):
            output = nn.forward(x)

            is_survived = 1 if output[0] > 0.5 else 0

            if is_survived == e[0]:
                correct_times += 1

        accuracy = correct_times / len(input_data)
        print(f"After Training Prediction Accuracy: {accuracy * 100: .2f} %")
        print()


class Task3:
    @staticmethod
    def run():
        print(f"{' Task 3: PyTorch Practice ':=^75}")

        print()
        print(f"{' Task 3 - 1 ':=^40}")
        t1 = torch.tensor([[2, 3, 1], [5, -2, 1]])
        print(f"{t1.shape = }, {t1.dtype = }")

        print()
        print(f"{' Task 3 - 2 ':=^40}")
        t2 = torch.rand(3, 4, 2)
        print(f"{t2.shape = }")
        print(t2)

        print()
        print(f"{' Task 3 - 3 ':=^40}")
        t3 = torch.ones(2, 1, 5)
        print(f"{t3.shape = }")
        print(t3)

        print()
        print(f"{' Task 3 - 3 ':=^40}")
        t4 = torch.tensor([[1, 2, 4], [2, 1, 3]])
        t5 = torch.tensor([[5], [2], [1]])
        print(f"{t4 @ t5 = }")

        print()
        print(f"{' Task 3 - 3 ':=^40}")
        t6 = torch.tensor([[1, 2], [2, 3], [-1, 3]])
        t7 = torch.tensor([[5, 4], [2, 1], [1, -5]])
        print(f"{t6 * t7 = }")


def main():
    Task1.run()
    Task2.run()
    Task3.run()


if __name__ == "__main__":
    main()

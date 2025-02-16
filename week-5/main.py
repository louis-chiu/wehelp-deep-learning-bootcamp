from network import Network
from loss_function import (
    BinaryCrossEntropy,
    MeanSqureError,
    LossFunction,
)


def regression_task():
    nn = (
        Network.builder()
        .add_fully_connected_layer(
            weight=[[0.5, 0.2], [0.6, -0.6]], bias=[0.3, 0.25], activation="relu"
        )
        .add_fully_connected_layer(weight=[[0.8, -0.5]], bias=[0.6])
        .add_fully_connected_layer(weight=[[0.6], [-0.3]], bias=[0.4, 0.75])
        .build()
    )
    input_ = [1.5, 0.5]
    expects = [0.8, 1]
    loss_fn: LossFunction = MeanSqureError()
    learning_rate = 0.01

    for epoch in range(1, 1001):
        outputs = nn.forward(input_)
        nn.backward(loss_fn.get_output_losses(outputs, expects))
        nn.zero_grad(learning_rate)
        if epoch == 1:
            print("=================== Task 1-1 ===================")

            for index, layer in enumerate(nn.layers):
                print(f"Layer {index}")
                print(layer.weight)
                print(layer.bias)
        elif epoch == 1000:
            print("=================== Task 1-2 ===================")
            loss = loss_fn.get_total_loss(outputs, expects)

            print(f"{loss=}")


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
    input_ = [0.75, 1.25]
    expects = [1]
    loss_fn: LossFunction = BinaryCrossEntropy()
    learning_rate = 0.01

    for epoch in range(1000):
        try:
            output = nn.forward(input_)
            nn.backward(loss_fn.get_output_losses(output, expects))
            nn.zero_grad(learning_rate)
            if epoch == 0:
                print("=================== Task 2-1 ===================")

                for index, layer in enumerate(nn.layers):
                    print(f"Layer {index}")
                    print(layer.weight)
                    print(layer.bias)
            elif epoch == 1000:
                print("=================== Task 2-2 ===================")
                loss = loss_fn.get_total_loss(output, expects)
                print(f"{loss=}")
        except Exception as e:
            print(f"=================== {type(e).__name__} ===================")
            print(f"output: {output}")
            print(f"ends at epoch: {epoch}")
            break


def main():
    regression_task()
    binary_classification_task()


if __name__ == "__main__":
    main()

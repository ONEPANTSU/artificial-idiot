from config import *
from dataset_handler import DatasetHandler
from network import Network


def main(file_path, sizes, batch_size, epochs_count, learning_rate):
    dataset_handler = DatasetHandler(file_path=file_path)
    dataset = dataset_handler.get_dataset()
    net = Network(sizes=sizes)

    net.dataset_testing(test_name="TESTS before SGD:", dataset=dataset)
    net.SGD(
        dataset=dataset,
        batch_size=batch_size,
        epochs_count=epochs_count,
        learning_rate=learning_rate,
    )
    net.dataset_testing(test_name="TESTS after SGD:", dataset=dataset)

    noisy_dataset = dataset_handler.add_noise(
        defective_percent=DEFECTIVE_PIXELS_COUNT_PERCENT
    )
    net.dataset_testing(test_name="TESTS with noise:", dataset=noisy_dataset)
    # for i in range(10):
    #     noisy_dataset = dataset_handler.add_noise(
    #         defective_percent=DEFECTIVE_PIXELS_COUNT_PERCENT
    #     )
    #     net.dataset_testing(test_name="TESTS with noise:", dataset=noisy_dataset)


if __name__ == "__main__":
    main(
        file_path=FILE_PATH,
        sizes=SIZES,
        batch_size=BATCH_SIZE,
        epochs_count=EPOCHS_COUNT,
        learning_rate=LEARNING_RATE,
    )

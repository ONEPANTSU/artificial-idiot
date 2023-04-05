import itertools
import math
import random

from config import *
from dataset_handler import DatasetHandler
from network import Network

def main(file_path, sizes, batch_size, epochs_count, learning_rate):
    dataset_handler = DatasetHandler(file_path=file_path)
    dataset = dataset_handler.get_full_dataset(DEFECTIVE_PIXELS_COUNT_PERCENT)

    net = Network(sizes=sizes)
    net.dataset_testing(test_name="TESTS before SGD:", dataset=dataset[TRAINING_DATASET_LENGTH:])
    net.sgd(
        dataset=dataset[:TRAINING_DATASET_LENGTH],
        batch_size=batch_size,
        epochs_count=epochs_count,
        learning_rate=learning_rate,
    )
    net.dataset_testing(test_name="TESTS after SGD:", dataset=dataset[TRAINING_DATASET_LENGTH:])



if __name__ == "__main__":
    main(
        file_path=FILE_PATH,
        sizes=SIZES,
        batch_size=BATCH_SIZE,
        epochs_count=EPOCHS_COUNT,
        learning_rate=LEARNING_RATE,
    )

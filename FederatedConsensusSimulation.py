import flwr as fl
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images to a range of [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# Define the CNN model
def create_cnn_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Flower client class
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.accuracy = None

    def get_parameters(self, config):
        print("Getting parameters from the client...")
        return self.model.get_weights()

    def fit(self, parameters, config):
        print("Starting training on the client...")
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        print("Training complete. Sending weights back to server...")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        print("Evaluating the model on the client...")
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        self.accuracy = accuracy
        print(f"Evaluation complete. Accuracy: {accuracy}")
        return loss, len(self.x_test), {"accuracy": accuracy}

    def submit_accuracy(self):
        return self.accuracy


class Participant:
    def __init__(self, id, stake):
        self.id = id
        self.stake = stake


class Miner(Participant):
    def __init__(self, id, stake, client):
        super().__init__(id, stake)
        self.client = client

    def submit_accuracy(self):
        return self.client.submit_accuracy()


class Validator(Participant):
    def __init__(self, id, stake):
        super().__init__(id, stake)
        self.miners = []
        self.local_average_accuracy = 0.0

    def assign_miner(self, miner):
        self.miners.append(miner)

    def aggregate_accuracies(self):
        total_accuracy = sum(miner.submit_accuracy() for miner in self.miners)
        self.local_average_accuracy = (
            total_accuracy / len(self.miners) if self.miners else 0
        )

    def get_aggregated_accuracy(self):
        return self.local_average_accuracy


class FederatedConsensus:
    def __init__(self, validators, miners):
        self.validators = validators
        self.miners = miners

    def distribute_miners(self):
        stake_ranges = [range(1, 11), range(11, 21), range(21, 31)]
        miner_groups = {stake_range: [] for stake_range in stake_ranges}

        # Group miners based on their stake ranges
        for miner in self.miners:
            for stake_range in stake_ranges:
                if miner.stake in stake_range:
                    miner_groups[stake_range].append(miner)

        # Distribute miners to validators ensuring diversity in stake ranges
        for stake_range, miners in miner_groups.items():
            for validator in self.validators:
                if miners:
                    validator.assign_miner(miners.pop(0))

    def collect_aggregated_accuracies(self):
        for validator in self.validators:
            validator.aggregate_accuracies()

    def print_clusters_and_accuracies(self):
        for validator in self.validators:
            print(f"{validator.id} cluster:")
            for miner in validator.miners:
                print(
                    f"  - {miner.id} (Stake: {miner.stake}, Accuracy: {miner.submit_accuracy()})"
                )
            print(f"  Average Accuracy: {validator.get_aggregated_accuracy()}\n")

    def select_best_validator(self):
        best_validator = max(self.validators, key=lambda v: v.get_aggregated_accuracy())
        print(
            f"Validator {best_validator.id} has the highest accuracy ({best_validator.get_aggregated_accuracy()})."
        )

    def run(self):
        self.distribute_miners()
        self.collect_aggregated_accuracies()
        self.print_clusters_and_accuracies()
        self.select_best_validator()


# Initialize Validators
validators = [Validator(id=f"Validator_{i+1}", stake=5 * (i + 1)) for i in range(4)]

# Initialize Miners by taking their stake as input and assigning a CIFAR-10 task
miners = []
client_data = []
num_clients = 5
partition_size = len(x_train) // num_clients

for i in range(num_clients):
    start = i * partition_size
    end = start + partition_size
    client_data.append((x_train[start:end], y_train[start:end]))

for i in range(num_clients):
    print(f"Enter details for Miner {i+1}:")
    stake = int(input(f"  Stake (1-30): "))
    model = create_cnn_model()
    client = CifarClient(model, client_data[i][0], client_data[i][1], x_test, y_test)

    # Run local training to get the accuracy
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())

    miner = Miner(id=f"Miner_{i+1}", stake=stake, client=client)
    miners.append(miner)

# Create the FederatedConsensus instance and run it
consensus = FederatedConsensus(validators=validators, miners=miners)
consensus.run()

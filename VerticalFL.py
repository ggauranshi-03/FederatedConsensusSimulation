import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train / 255.0, -1)
x_test = np.expand_dims(x_test / 255.0, -1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


def create_cnn_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
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
class VFLClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.accuracy = None

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        self.accuracy = accuracy
        return loss, len(self.x_test), {"accuracy": accuracy}

    def submit_accuracy(self):
        return self.accuracy


class VFLParticipant:
    def __init__(self, id, stake):
        self.id = id
        self.stake = stake


class VFLMiner(VFLParticipant):
    def __init__(self, id, stake, client):
        super().__init__(id, stake)
        self.client = client

    def submit_accuracy(self):
        return self.client.submit_accuracy()


class VFLValidator(VFLParticipant):
    def __init__(self, id, stake):
        super().__init__(id, stake)
        self.miners = []
        self.last_miner_states = {}
        self.model_aggregator = None

    def assign_miner(self, miner):
        self.miners.append(miner)

    def set_model_aggregator(self, aggregator):
        self.model_aggregator = aggregator

    def aggregate_updates(self):
        if self.model_aggregator:
            # Collect and aggregate model updates from miners
            model_updates = [miner.client.get_parameters(None) for miner in self.miners]
            aggregated_params = self.model_aggregator.aggregate(model_updates)
            return aggregated_params
        return None

    def evaluate_and_update(self, parameters):
        total_loss = 0
        total_accuracy = 0
        for miner in self.miners:
            loss, _, metrics = miner.client.evaluate(parameters, None)
            total_loss += loss
            total_accuracy += metrics["accuracy"]
            print(f"Miner {miner.id} accuracy: {metrics['accuracy']}")
        avg_accuracy = total_accuracy / len(self.miners) if self.miners else 0
        return avg_accuracy

    def handle_miner_departure(self, miner_id, last_state):
        # If a miner leaves, use the last state of the departing miner to handle its tasks
        self.last_miner_states[miner_id] = last_state

    def get_last_miner_state(self, miner_id):
        return self.last_miner_states.get(miner_id, None)

    def print_cluster(self):
        print(f"Validator {self.id} cluster:")
        for miner in self.miners:
            print(
                f"  - Miner {miner.id} (Stake: {miner.stake}, Accuracy: {miner.client.submit_accuracy()})"
            )


class VFLModelAggregator:
    def aggregate(self, model_updates):
        avg_params = [
            np.mean(np.array(params), axis=0) for params in zip(*model_updates)
        ]
        return avg_params


class VFLConsensus:
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

    def run(self, epochs=10, rounds=1):
        # Initialize model aggregator for validators
        model_aggregator = VFLModelAggregator()

        for validator in self.validators:
            validator.set_model_aggregator(model_aggregator)

        for round in range(rounds):
            print(f"Round {round + 1}")

            self.distribute_miners()

            for validator in self.validators:
                # Initialize parameters for this round
                aggregated_params = None
                for epoch in range(epochs):
                    print(f"  Epoch {epoch + 1}")

                    # Aggregate updates from miners
                    aggregated_params = validator.aggregate_updates()
                    avg_accuracy = validator.evaluate_and_update(aggregated_params)
                    print(
                        f"  Validator {validator.id} average accuracy: {avg_accuracy}"
                    )

                # Print cluster details
                validator.print_cluster()

            # Optionally, select the best validator based on accuracy
            best_validator = max(
                self.validators,
                key=lambda v: v.evaluate_and_update(v.aggregate_updates()),
            )
            print(f"Best Validator: {best_validator.id}")


# Initialize Validators
validators = [VFLValidator(id=f"Validator_{i+1}", stake=5 * (i + 1)) for i in range(4)]

# Initialize Miners by taking their stake as input and assigning an MNIST task
miners = []
client_data = []
num_clients = 15  # Example number of clients
partition_size = len(x_train) // num_clients

# Vertically partition the features
for i in range(num_clients):
    start = i * partition_size
    end = start + partition_size
    client_data.append((x_train[start:end], y_train[start:end]))

for i in range(num_clients):
    print(f"Enter details for Miner {i+1}:")
    stake = int(input(f"  Stake (1-30): "))
    model = create_cnn_model()
    client = VFLClient(model, client_data[i][0], y_train, x_test, y_test)

    miner = VFLMiner(id=f"Miner_{i+1}", stake=stake, client=client)
    miners.append(miner)

# Create the VFLConsensus instance and run it
consensus = VFLConsensus(validators=validators, miners=miners)
consensus.run(epochs=10, rounds=1)

import logging

import flwr as fl
from flwr.server.strategy import FedAvg

# Set logging level to DEBUG for more detailed output
logging.basicConfig(level=logging.DEBUG)
strategy = FedAvg(
    min_fit_clients=1,  # Minimum number of clients to be sampled for the next round
    min_available_clients=1,  # Minimum number of clients that need to be available for training
    min_evaluate_clients=1,
)  # Minimum number of clients to be sampled for evaluation
# Create a ServerConfig object with the desired number of rounds
config = fl.server.ServerConfig(num_rounds=1)

fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
)

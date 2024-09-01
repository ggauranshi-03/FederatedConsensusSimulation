import logging

import flwr as fl
from flwr.server.strategy import FedAvg

logging.basicConfig(level=logging.DEBUG)
strategy = FedAvg(
    min_fit_clients=1,
    min_available_clients=1,
    min_evaluate_clients=1,
)
config = fl.server.ServerConfig(num_rounds=1)

fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
)

# server.py

import flwr as fl
from flwr.server.strategy import FedAvg


def main():
    strategy = FedAvg(
        min_fit_clients=4,
        min_available_clients=4,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8000",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

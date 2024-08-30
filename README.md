# Federated Learning with Flower Framework

This project demonstrates how to use the Flower (`flwr`) framework to implement federated learning with TensorFlow on the CIFAR-10 dataset. Federated learning allows multiple clients (referred to as "Miners" in this project) to train models on their local data without sharing the data with a central server. The server (using the Flower framework) aggregates the results from the clients to create a global model.

## Key Components

### 1. Dataset Loading and Preprocessing
- The CIFAR-10 dataset is loaded using TensorFlow's `cifar10.load_data()` function. The dataset consists of 60,000 32x32 color images in 10 classes, with 50,000 training images and 10,000 test images.
- The images are normalized to a range of [0, 1] by dividing by 255.0.
- The labels are converted to one-hot encoded format using `to_categorical()`.

### 2. CNN Model Definition
- The `create_cnn_model()` function defines a Convolutional Neural Network (CNN) with three convolutional layers followed by max pooling, flattening, and dense layers.
- The model is compiled using the Adam optimizer and categorical cross-entropy loss, with accuracy as a metric.

### 3. Flower Client (`CifarClient`)
- The `CifarClient` class extends `fl.client.NumPyClient`, which is a base class for Flower clients that use NumPy arrays for training data.
- The client class has the following key methods:
  - `get_parameters`: Returns the current weights of the model.
  - `fit`: Trains the model on the local data (`x_train`, `y_train`) and returns the updated weights.
  - `evaluate`: Evaluates the model on the local test data (`x_test`, `y_test`) and returns the loss and accuracy.
  - `submit_accuracy`: Returns the accuracy obtained after the evaluation.

### 4. Participants, Miners, and Validators
- The `Participant` class is a base class representing participants in the federated learning process.
- `Miner` inherits from `Participant` and is associated with a Flower client (`CifarClient`). Each miner has a `stake` (a numerical value used to distribute miners among validators) and can submit its model accuracy.
- `Validator` also inherits from `Participant` and can be assigned miners. It aggregates the accuracies of its assigned miners to calculate a local average accuracy.
- `FederatedConsensus` orchestrates the distribution of miners to validators based on their stakes, aggregates the accuracies from validators, and selects the best validator based on the highest aggregated accuracy.

### 5. Federated Learning Workflow
- A list of `Validator` objects is created, each with a different stake.
- The training data is partitioned among the miners, and each miner is assigned a unique portion of the dataset.
- Each miner trains a model using its local data and reports the accuracy.
- The `FederatedConsensus` object manages the distribution of miners to validators, aggregates accuracies, and selects the best validator.

### 6. Server Setup and Federated Learning Execution
- A Flower server is started using the `fl.server.start_server()` function. This server aggregates the results from the clients (miners) using a federated averaging strategy (`FedAvg`).
- The server configuration specifies the number of rounds (in this case, 1) and the strategy to use for aggregation.
- The Flower clients connect to this server, train their models, and send the updated weights back to the server for aggregation.

## How Flower Works in This Project

Flower enables the federated learning process by providing a framework for distributed model training. It abstracts away the complexity of client-server communication and allows easy implementation of federated learning across multiple clients.

Each client (Miner) trains a local model on its subset of the data, and the server (using the Flower framework) aggregates these models to produce a global model. The federated learning process in this project mimics a real-world scenario where data is distributed across multiple devices, and a central server coordinates the training process without needing access to the data on the clients.

## Workflow

1. **Initialization**:
   - The CIFAR-10 data is loaded, preprocessed, and partitioned among the clients.
   - The CNN model is defined and compiled.

2. **Client Training**:
   - Each client (Miner) trains its model on local data for one epoch.
   - The client then evaluates the model and submits the accuracy.

3. **Aggregation and Consensus**:
   - Validators aggregate the accuracies of their assigned miners.
   - The consensus mechanism selects the best validator based on the aggregated accuracy.

4. **Server-Side Aggregation**:
   - The Flower server uses the FedAvg strategy to aggregate the weights from all clients and updates the global model.

This project provides a basic implementation of federated learning using Flower, demonstrating how multiple clients can collaborate to train a global model without sharing their local data.

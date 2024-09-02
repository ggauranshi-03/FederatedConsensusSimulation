from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Bin the 'Age' feature
def _bin_age(age_series):
    bins = [-np.inf, 10, 40, np.inf]
    labels = ["Child", "Adult", "Elderly"]
    return (
        pd.cut(age_series, bins=bins, labels=labels, right=True)
        .astype(str)
        .replace("nan", "Unknown")
    )


# Extract titles from the 'Name' feature
def _extract_title(name_series):
    titles = name_series.str.extract(" ([A-Za-z]+)\.", expand=False)
    rare_titles = {
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    }
    titles = titles.replace(list(rare_titles), "Rare")
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return titles


# Create features and process the data
def _create_features(df):
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = _bin_age(df["Age"])
    df["Cabin"] = df["Cabin"].str[0].fillna("Unknown")
    df["Title"] = _extract_title(df["Name"])
    df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)
    all_keywords = set(df.columns)
    df = pd.get_dummies(
        df, columns=["Sex", "Pclass", "Embarked", "Title", "Cabin", "Age"]
    )
    return df, all_keywords


# Partition data into different feature groups
def _partition_data(df, all_keywords):
    partitions = []
    keywords_sets = [{"Parch", "Cabin", "Pclass"}, {"Sex", "Title"}]
    keywords_sets.append(all_keywords - keywords_sets[0] - keywords_sets[1])

    for keywords in keywords_sets:
        partitions.append(
            df[
                list(
                    {
                        col
                        for col in df.columns
                        for kw in keywords
                        if kw in col or "Survived" in col
                    }
                )
            ]
        )

    return partitions


# Get partitions and label data
def get_partitions_and_label():
    df = pd.read_csv("./train.csv")
    processed_df = df.dropna(subset=["Embarked", "Fare"]).copy()
    processed_df, all_keywords = _create_features(processed_df)
    raw_partitions = _partition_data(processed_df, all_keywords)

    partitions = []
    for partition in raw_partitions:
        partitions.append(partition.drop("Survived", axis=1))
    return partitions, processed_df["Survived"].values


# Main logic to split the data
partitions, y = get_partitions_and_label()

# Split the data into training and testing sets
x_train_parts = []
x_test_parts = []
for partition in partitions:
    x_train, x_test, y_train, y_test = train_test_split(
        partition, y, test_size=0.2, random_state=42
    )
    x_train_parts.append(x_train)
    x_test_parts.append(x_test)

# Initialize Miners and Clusters
num_clusters = 4
miners_per_cluster = 3
total_miners = num_clusters * miners_per_cluster


class VFLClient:
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.accuracy = None

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.accuracy = accuracy
        return accuracy

    def submit_accuracy(self):
        return self.accuracy


class Miner:
    def __init__(self, id, stake, client):
        self.id = id
        self.stake = stake
        self.client = client

    def submit_accuracy(self):
        return self.client.submit_accuracy()


class Validator:
    def __init__(self, id, stake):
        self.id = id
        self.stake = stake
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

        for miner in self.miners:
            for stake_range in stake_ranges:
                if miner.stake in stake_range:
                    miner_groups[stake_range].append(miner)

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
validators = [
    Validator(id=f"Validator_{i+1}", stake=5 * (i + 1)) for i in range(num_clusters)
]

# Initialize Miners
miners = []
for i in range(total_miners):
    stake = int(input(f"Enter stake for Miner {i+1} (1-30): "))
    model = LogisticRegression(max_iter=500)

    partition_idx = i % len(x_train_parts)

    client = VFLClient(
        model,
        x_train_parts[partition_idx],
        y_train,
        x_test_parts[partition_idx],
        y_test,
    )

    # Train the model for each miner's client
    client.fit()
    client.evaluate()

    miner = Miner(id=f"Miner_{i+1}", stake=stake, client=client)
    miners.append(miner)

# Create the FederatedConsensus instance and run it
consensus = FederatedConsensus(validators=validators, miners=miners)
consensus.run()

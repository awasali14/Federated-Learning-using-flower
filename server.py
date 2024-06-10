import flwr as fl
import sys
import numpy as np

# Define custom aggregation functions
def aggregate_fit_metrics(metrics):
    aggregated_metrics = {}
    for metric_tuple in metrics:
        for key, value in metric_tuple[1].items():  # metric_tuple[1] is the actual metrics dictionary
            if key not in aggregated_metrics:
                aggregated_metrics[key] = []
            aggregated_metrics[key].append(value)
    aggregated_metrics = {key: np.mean(values) for key, values in aggregated_metrics.items()}
    return aggregated_metrics

def aggregate_evaluate_metrics(metrics):
    aggregated_metrics = {}
    for metric_tuple in metrics:
        for key, value in metric_tuple[1].items():  # metric_tuple[1] is the actual metrics dictionary
            if key not in aggregated_metrics:
                aggregated_metrics[key] = []
            aggregated_metrics[key].append(value)
    aggregated_metrics = {key: np.mean(values) for key, values in aggregated_metrics.items()}
    return aggregated_metrics

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

# Create strategy and run server
strategy = SaveModelStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    fit_metrics_aggregation_fn=aggregate_fit_metrics,
    evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
)

# Start Flower server for three rounds of federated learning
fl.server.start_server(
    server_address='localhost:' + str(sys.argv[1]),
    config=fl.server.ServerConfig(num_rounds=10),
    grpc_max_message_length=1024*1024*1024,
    strategy=strategy
)

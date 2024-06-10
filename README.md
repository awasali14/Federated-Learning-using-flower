# Federated Learning with Flower and TensorFlow

## Project Overview

This project demonstrates the implementation of federated learning using the Flower framework and TensorFlow. The goal is to simulate a federated learning environment where multiple clients train a shared model on their local data and then aggregate their results on a central server.

## Motivation

Exploring decentralized machine learning approaches that preserve data privacy is the key motivation behind this project. Federated learning enables multiple clients to collaboratively train a model without sharing their raw data, addressing privacy concerns in a distributed training setting.

## Problem Addressed

The project addresses the challenge of training a machine learning model in a distributed manner while ensuring data privacy. Traditional centralized training methods require aggregating data from multiple sources, which may not be feasible due to privacy concerns or data sensitivity. Federated learning provides a solution by enabling decentralized training.

## What Was Learned

- Gained an understanding of federated learning concepts and architecture.
- Acquired hands-on experience with the Flower framework and TensorFlow for federated learning.
- Implemented custom aggregation strategies for federated learning.
- Managed non-IID (non-Independent and Identically Distributed) data across clients.

## Project Structure

### Server

The server coordinates the federated learning process, managing the communication and aggregation of model updates from clients. A custom strategy was implemented to save aggregated model weights after each round.

### Clients

Five clients were set up, each with a unique data distribution to simulate non-IID data scenarios. Each client trains a local model on its subset of data and sends the updates to the server.

### Data Distribution

- **Client 1**: Mostly class 0 and 1.
- **Client 2**: Mostly class 2 and 3.
- **Client 3**: Mostly class 4 and 5.
- **Client 4**: Mostly class 6 and 7.
- **Client 5**: Mostly class 8 and 9.

### Training Process

1. **Initialize Server**: The server starts and waits for clients to connect.
2. **Client Connection**: Clients connect to the server and request initial model parameters.
3. **Local Training**: Each client trains the model on its local data and sends the updates back to the server.
4. **Aggregation**: The server aggregates the updates from clients and updates the global model.
5. **Evaluation**: The server evaluates the aggregated model to track performance.
6. **Repeat**: Steps 3-5 are repeated for multiple rounds.

### Results Summary

- The training accuracy improved significantly across rounds.
- Initial rounds showed varied accuracy due to non-IID data.
- Final rounds achieved higher accuracy, demonstrating effective federated learning.

## How to Run

### Prerequisites

- Python 3.8+
- TensorFlow
- Flower

### Steps to Run the Server

1. Navigate to the project directory.
2. Start the server using the following command:
   ```sh
   python server.py <PORT>
   ```

### Steps to Run the Clients

1. Open multiple terminal windows (one for each client).
2. In each terminal, navigate to the project directory and start the client using the following commands:
   ```sh
   python client1.py <PORT>
   python client2.py <PORT>
   python client3.py <PORT>
   python client4.py <PORT>
   python client5.py <PORT>
   ```

### Example Commands

```sh
# Terminal 1 (Server)
python server.py 5002

# Terminal 2 (Client 1)
python client1.py 5002

# Terminal 3 (Client 2)
python client2.py 5002

# Terminal 4 (Client 3)
python client3.py 5002

# Terminal 5 (Client 4)
python client4.py 5002

# Terminal 6 (Client 5)
python client5.py 5002
```

By following these steps, the federated learning setup will be executed, with the server coordinating the training process across multiple clients.

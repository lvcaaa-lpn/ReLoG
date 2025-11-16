# Review Locally, Recommend Globally

Our work proposes a novel privacy-preserving recommendation system that leverages federated learning (FL). The core idea is to allow users to contribute to a global recommendation model without ever sharing their raw textual reviews.

- **Local Processing**: User reviews are transformed into semantic embeddings on the user's device using a public, pre-trained model (e.g., Sentence-BERT).
- **Privacy by Design**: The raw text never leaves the device and can be discarded immediately after encoding.
- **Federated Learning**: A lightweight global model (XGBoost or a compact MLP) is trained collaboratively on these user-generated embeddings. Only processed, ephemeral information like model updates or aggregated statistics are shared with the server.

This approach combines the rich semantic information from natural language with the privacy guarantees of federated learning, proving effective even in cold-start scenarios.

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Running the Experiments](#running-the-experiments)
  - [Main Experiments (Ablation Study)](#1--main-experiments-ablation-study)
  - [Cold-Start Evaluation](#2--cold-start-evaluation)
- [Code Overview](#code-overview)
- [Citation](#citation)

## Getting Started

1.  **Install the required dependencies:**
    The `requirements.txt` file lists all necessary libraries. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download the datasets:**
    The scripts require the **Amazon Review Data**, specifically the "Amazon Fashion" and "All Beauty" subsets.
    - Download the review files (`Amazon_Fashion.json.gz`, `All_Beauty.json.gz`) and metadata files (`meta_Amazon_Fashion.json.gz`, `meta_All_Beauty.json.gz`).
    - **Place the `.gz` files in the root directory of this project.**

## Running the Experiments

The repository includes two main scripts.

### 1. Main Experiments (Ablation Study)

The `ablation.py` script performs data preprocessing, trains the ReLoG model with different configurations (embedders and classifiers), and evaluates its performance.

To run this script, simply execute:
```bash
python ablation.py
```
>[!NOTE]
>The script name in your repository might be different. Please adjust the command accordingly.

The script will print the final performance metrics to the console.

### 2. Cold-Start Evaluation

The `coldstart.py` script creates three disjoint test sets to simulate different scenarios:
- **Warm-Start**: Known users interacting with known items.
- **User Cold-Start**: New users interacting with known items.
- **Item Cold-Start**: Known users interacting with new items.

To run the cold-start evaluation, execute:
```bash
python coldstart.py
```
The script will train the model and then report the AUC and F1-score for each of the three scenarios in a comparative table.

## Code Overview

-   `ablation.py`: this is the core implementation of our framework. It contains the full data processing pipeline, feature construction logic, and the training/evaluation loops for the models. This file is used to generate the main results and ablation studies.

-   `coldstart.py`: this script adapts the main pipeline to specifically test the model's robustness in cold-start scenarios. It handles the logic for splitting the data into "warm", "user-cold", and "item-cold" sets to provide a detailed performance breakdown under these challenging conditions.

-   `requirements.txt`: a file listing all the Python libraries and their specific versions needed to run the code, ensuring reproducibility.

# Review Locally, Recommend Globally (ReLoG)
 
ReLoG is a **federated** recommendation system that combines rich text embeddings with the privacy guarantees of federated learning. The central idea is to allow each user to contribute to a global model without ever sharing the raw text of their reviews: only the shared network weights are exchanged with the server.
 
- **Local processing**: User reviews and item metadata are transformed into semantic embeddings using a pre-trained SBERT model (`all-MiniLM-L6-v2`, 384 dimensions).
- **Two-tower architecture**: a **User Tower** and an **Item Tower**, shared and aggregated across clients, project the embeddings into a common, L2-normalized latent space. A **local scoring function** (`client_mlp`), specific to each user and maintained only on the client, estimates user-item relevance by combining the two representations.
- **Federated training**: follows the **Federated Averaging with momentum** paradigm—in each round, a subset of users trains their own towers locally (using **BPR loss** and **hard negative sampling**), and only the weights of the shared towers are aggregated globally; the `client_mlp` remains local and is never aggregated.
- **Two-level evaluation**: **warm users** (seen during training) are evaluated using **HR@k** and **NDCG@k** in a leave-one-out protocol, following a brief local fine-tuning. **Unseen users**, on the other hand, are evaluated in **few-shot** mode (1-, 2-, 3-shot, or “full”) to measure the model’s ability to adapt to new users with very few interactions.

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Data](#data)
- [Running the Experiment](#running-the-experiment)
- [Notebook Structure](#notebook-structure)
- [Output](#output)
## Getting Started
 
### Prerequisites
Install the necessary dependencies:
```bash
pip install -r requirements.txt
```
The main libraries used are: `pandas`, `numpy`, `torch`, `sentence-transformers`, `tqdm`.
 
If you have a GPU available, the notebook will use it automatically (`device = ‘cuda’ if torch.cuda.is_available() else ‘cpu’`); otherwise, it will run on the CPU as well, but more slowly (especially when calculating SBERT embeddings).

### Data
The notebook expects two pre-processed **Parquet** files:
- `electronics_review.parquet` — user-item interactions (reviews), with columns such as `user_id_int`, `item_id_int`, `timestamp`, `summary`, and `reviewText`.
- `electronics_meta.parquet` — item metadata, with a `meta_text` column.
In the notebook, these files are loaded using a relative path:
```python
df_sampled      = pd.read_parquet(‘../../preprocessing/electronics_review.parquet’)
df_meta_aligned = pd.read_parquet(‘../../preprocessing/electronics_meta.parquet’)
```
Update these paths based on where you store your preprocessed data.
 
## Running the Experiment
Unlike a pipeline with separate scripts, the entire workflow (preprocessing, federated training, evaluation) is contained in a **single Jupyter notebook**: `ReLoG_dif_seed__1_.ipynb`.
 
To run it:
1. Open the notebook and run the cells in order from top to bottom.
2. The main hyperparameters (learning rate, number of global rounds, clients per round, k for metrics, etc.) are defined within the `run_experiment` function and can be modified directly there.
3. In the last cell, you can set the seeds to use when repeating the experiment:
```python
   seeds = [0]  # 0, 1, 2, 3, 4
```
 By default, only one seed is configured. To obtain aggregate statistics (mean ± standard deviation) across multiple runs, as indicated by the final message displayed on the screen, add more seeds to the list, for example, `seeds = [0, 1, 2, 3, 4]`.

## Notebook Structure
The notebook is organized into the following sections:
- **Import and setup**: loading libraries and defining `set_seed` for reproducibility.
- **Loading data and calculating embeddings**: reading Parquet files and encoding reviews/metadata with SBERT.
- **Architecture definition**: `UserTower`, `ItemTower`, `LocalScoreFunction`, and the complete `TwoTowerRecommender` model, plus `bpr_loss`.
- **Data utilities**: `get_client_data`, which constructs the train/val/test split for a single user based on the timestamp.
- **Hard negative sampling**: `sample_hard_negatives`, to select “hard” negatives that are more informative than random negatives.
- **Local client training**: `train_client`, with linear warmup + cosine annealing of the learning rate.
- **Federated aggregation**: `weighted_fedavg_momentum`, FedAvg weighted by the number of interactions, with momentum.
- **Warm user evaluation**: `evaluate_top_k`, leave-one-out protocol with brief local fine-tuning.
- **Few-shot evaluation (unseen users)**: `get_fewshot_data` and `evaluate_fewshot`, to measure adaptability to new users.
- **User split**: `split_users`, divides warm and unseen users in a reproducible manner given a seed.
- **Experiment orchestration**: `run_experiment`, performs a complete run (federated training + selection of the best model on validation + final test) for a given seed.
- **Multi-seed execution**: Repeats `run_experiment` using the configured seeds and aggregates the results.


## Output
For each seed run, the notebook prints:
- **HR@K** and **NDCG@K** for warm users (last interaction, leave-one-out).
- **HR@K** and **NDCG@K** for unseen users, in the **1-shot**, **2-shot**, **3-shot**, and **full** regimes.
Upon completion, if multiple seeds were run, a summary table is printed showing the **mean ± standard deviation** for each scenario.

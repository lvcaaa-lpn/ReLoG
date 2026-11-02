# ReLoG: Review Locally, Recommend Globally

Code for the paper *Review Locally, Recommend Globally: Private Recommendations from Your Own Reviews*
(A. Mora, L. Napoli, P. Bellavista, University of Bologna).

ReLoG is a federated recommender in which each user is a client. A frozen Sentence-BERT encoder
(`all-MiniLM-L6-v2`) embeds the user's reviews and the item metadata on the device. A user tower and an
item tower, shared across clients, project these embeddings into a common space and are trained with
FedAvg and server momentum. A small scoring MLP stays on each client and is never sent to the server.
Training uses a BPR loss with hard negatives.

## Repository

```
ReLoG.ipynb                        training and evaluation of ReLoG, optional user-level DP, server-side attacks
ablation/ReLoG_ablations.ipynb     dot product, tower depth, synthetic reviews (template / SLM)
ablation/ReLoG_ID_clean.ipynb      learned ID embeddings instead of SBERT inputs
ablation/generate_slm_reviews.ipynb  generation of the SLM reviews used by the `slm` ablation
preprocessing/preprocessing.ipynb  construction of the datasets from Amazon Review Data 2018
```

## Setup

```bash
pip install -r requirements.txt
```

A GPU is recommended. `ollama` is only needed to generate the SLM reviews.

## Data

The experiments use three categories of Amazon Review Data (2018): Movies and TV, CDs and Vinyl, and
Electronics. Run `preprocessing/preprocessing.ipynb` once per category (set `CATEGORY`): it downloads the
5-core review file and the metadata, applies 5-core filtering, samples users up to about 50,000 interactions
and writes `<dataset>_review.parquet` and `<dataset>_meta.parquet` to `preprocessing/`.

| Dataset | Users | Items | Interactions |
|---|---|---|---|
| Movies and TV | 4,376 | 19,199 | 50,600 |
| CDs and Vinyl | 3,905 | 27,149 | 49,659 |
| Electronics | 5,439 | 27,845 | 49,202 |

## Running

Open `ReLoG.ipynb`, set the configuration cell and run all cells.

- `DATASET`: `movies`, `cd` or `electronics`.
- `SEEDS`: the paper reports mean and standard deviation over seeds 0-4.
- `CONFIGS`: `dict(mode='none')` for ReLoG without DP; `dict(mode='central', z=...)` for DP-A (noise on the
  securely aggregated sum); `dict(mode='local', z=...)` for DP-B (noise added by each client).
  The paper uses `z` = 1 and 3 for DP-A and `z` = 5 and 10 for DP-B, with `q = 0.05` and `delta = 1e-5`.
- `ATTACK_EVERY`: set to 5 to run the membership-inference and review-leakage attacks on the updates
  observed by the server.

Protocol: 20% of the users are held out as unseen users. For the others, the last interaction is the test
target and the second to last the validation target, used to select the checkpoint. Unseen users are
evaluated after adapting the local MLP on their first 1, 2 or 3 interactions, or on their full history.
Items are ranked against the whole catalogue and HR@20 and NDCG@20 are reported.

The ablation notebooks follow the same procedure; select the variant with `ABLATION`.

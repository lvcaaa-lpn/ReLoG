# ==============================================================================
# 1. IMPORTS AND EXPERIMENT CONFIGURATION
# ==============================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.spatial.distance import cosine
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import time 
import os   
from sklearn.utils.class_weight import compute_class_weight
import gc


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

USER_INTERACTION_THRESHOLDS = [
    # 3,
     5,
    # 10
    ]

# --- EMBEDDING MODELS ---
EMBEDDING_MODELS_TO_TEST = [
    'all-mpnet-base-v2',        
    # 'all-MiniLM-L6-v2',         
]

# --- CLASSIFIER ---
CLASSIFIERS_TO_TEST = [
    'xgboost',
    # 'mlp_centralized',  
    # 'mlp_federated'     
]


# ==============================================================================
# 2. DATA PREPARATION FUNCTIONS
# ==============================================================================
CHUNKSIZE = 50000

def load_and_preprocess_data(review_path, meta_path, sample_frac=None, min_user_interactions=5, min_item_interactions=3):
     # The process is carried out in 3 phases:
    # 1. PHASE 1: Identify active users (>= min_user_interactions) based on the entire dataset.
    # 2. PHASE 2: Calculate product counts considering ONLY the interactions of users
    #    identified in Pass 1. Identify active products (>= min_item_interactions).
    # 3. PHASE 3: Build the final DataFrame using the stable sets of users and products.

    print(f"\n[START] Preprocessing with sequential filtering on " + review_path)
    print(f"  - Criteria: Users  >= {min_user_interactions}, Prodotti >= {min_item_interactions}")
    if sample_frac:
        print(f"--- SAMPLING MODE ACTIVE (fraction={sample_frac}) ---")

    print("\n  - Loading and initial metadata filtering...")
    meta = pd.read_json(meta_path, lines=True, compression='gzip')
    meta = meta[['parent_asin', 'title', 'description', 'price', 'categories']]
    meta.columns = ['product_id', 'title', 'description', 'price', 'categories']
    meta.dropna(subset=['title'], inplace=True)
    meta_filtered_title = meta[meta['title'].str.len() > 10].copy()
    valid_products_by_title = set(meta_filtered_title['product_id'])

    # --- PHASE 1: IDENTIFY ACTIVE USERS ---
    print("\n[FASE 1] Passata 1: Identificazione degli utenti attivi...")
    user_counts = Counter()
    review_iterator = pd.read_json(review_path, lines=True, compression='gzip', chunksize=CHUNKSIZE)
    for i, chunk in enumerate(review_iterator):
        print(f"\r    - Conteggio utenti nel blocco {i+1}...", end="")
        user_counts.update(chunk['user_id'])
    print("\n  - Conteggio utenti completato.")

    active_users = {user for user, count in user_counts.items() if count >= min_user_interactions}
    print(f"  - Identificati {len(active_users)} utenti attivi.")

    # --- PHASE 2: IDENTIFY ACTIVE PRODUCTS (based on active users) ---
    print("\n[FASE 2]  PHASE 2: IDENTIFY ACTIVE PRODUCTS (based on PHASE 1 users)...")
    product_counts = Counter()
    review_iterator = pd.read_json(review_path, lines=True, compression='gzip', chunksize=CHUNKSIZE)
    for i, chunk in enumerate(review_iterator):
        print(f"\r    - Counting products in chunk {i+1}...", end="")
        chunk_filtered_users = chunk[chunk['user_id'].isin(active_users)]
        product_counts.update(chunk_filtered_users['asin'])
    print("\n  - Product count completed.")

    active_products = {prod for prod, count in product_counts.items() if count >= min_item_interactions}
    print(f"  - Identified {len(active_products)} active products.")
    del product_counts
    gc.collect()

    final_active_products = active_products.intersection(valid_products_by_title)
    print(f"  - {len(final_active_products)} products remain after title filter.")

        # --- SAMPLING PHASE ---
    if sample_frac is not None and 0 < sample_frac < 1:
        print("\n[FASE X] Performing stratified sampling...")

        active_user_counts_df = pd.DataFrame(
            [(user, count) for user, count in user_counts.items() if user in active_users],
            columns=['user_id', 'review_count']
        ).set_index('user_id')

        try:
            strata = pd.qcut(active_user_counts_df['review_count'], q=4, labels=False, duplicates='drop')
        except ValueError:
            print("    - Warning: could not create 4 strata, falling back to 2.")
            strata = pd.qcut(active_user_counts_df['review_count'], q=2, labels=False, duplicates='drop')

        sampled_user_ids = strata.groupby(strata).apply(
            lambda x: x.sample(frac=sample_frac, random_state=42)
        ).index.get_level_values(1)

        final_user_set = set(sampled_user_ids)
        print(f"    - Users before sampling: {len(active_users)}. Sampled users: {len(final_user_set)}.")
    else:
        # If no sampling, use all active users
        final_user_set = active_users

    del user_counts 
    gc.collect()

    # --- PHASE 3: BUILD FINAL DATAFRAME ---
    print("\n[FASE 3] PHASE 3: Building the final DataFrame...")
    filtered_chunks = []
    review_iterator = pd.read_json(review_path, lines=True, compression='gzip', chunksize=CHUNKSIZE)
    for i, chunk in enumerate(review_iterator):
        print(f"\r    - Filtering chunk {i+1}...", end="")

        chunk = chunk[chunk['user_id'].isin(final_user_set)]
        chunk = chunk[chunk['asin'].isin(final_active_products)]

        if chunk.empty:
            continue

        chunk = chunk[['user_id', 'asin', 'text', 'rating', 'timestamp']] 
        chunk.rename(columns={'asin': 'product_id', 'text': 'review_text'}, inplace=True)

        merged_chunk = pd.merge(chunk, meta_filtered_title, on='product_id', how='inner')
        if not merged_chunk.empty:
            filtered_chunks.append(merged_chunk)

    print("\n  - Concatenating final chunks...")
    if not filtered_chunks:
        raise ValueError("No data left after filtering.")

    df_filtered = pd.concat(filtered_chunks, ignore_index=True)

    # --- Final statistics ---
    print("\n" + "="*50)
    print("      FINAL DATASET STATISTICS     ")
    print("="*50)
    final_users = df_filtered['user_id'].nunique()
    final_reviews = len(df_filtered)
    print(f"  - Total Number of Reviews: {final_reviews}")
    print(f"  - Unique Users: {final_users}")
    print(f"  - Unique Products: {df_filtered['product_id'].nunique()}")
    print(f"  - Average Reviews per User: {final_reviews / final_users:.2f}")
    print("="*50 + "\n")

    return df_filtered

def split_data_by_user(df):
    # Splits the dataset into train, validation, and test sets, keeping users disjoint.
    print("[PHASE 1] Splitting users into train/val/test...")
    all_users = df['user_id'].unique()
    train_val_users, test_users = train_test_split(all_users, test_size=0.15, random_state=42)
    train_users, val_users = train_test_split(train_val_users, test_size=(0.10/0.85), random_state=42)

    train_data = df[df['user_id'].isin(train_users)].copy()
    val_data = df[df['user_id'].isin(val_users)].copy()
    test_data = df[df['user_id'].isin(test_users)].copy()

    print(f"Users: {len(train_users)} train, {len(val_users)} validation, {len(test_users)} test.")
    return train_data, val_data, test_data
    
def generate_embeddings(sbert_model, model_name_str, train_df, val_df, test_df):
    # Computes embeddings for reviews and products.
    print(f"[PHASE 2] Computing embeddings with model '{model_name_str}'...")

    # Review embeddings
    for df, name in [(train_df, "Train"), (val_df, "Validation"), (test_df, "Test")]:
        print(f"  - Computing review embeddings for the {name} Set...")
        texts = df['review_text'].tolist()
        df['review_emb'] = sbert_model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    # Product embeddings (computed only on the training set to prevent data leakage)
    print("  - Computing product embeddings...")
    train_df.loc[:,'description'] = train_df['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    train_df.loc[:,'product_text'] = train_df['title'].fillna('') + ' ' + train_df['description'].fillna('')

    product_texts = train_df.drop_duplicates('product_id')[['product_id', 'product_text']]
    product_embs = sbert_model.encode(product_texts['product_text'].tolist(), batch_size=64, show_progress_bar=True)
    product_profiles = dict(zip(product_texts['product_id'], product_embs.tolist()))

    return train_df, val_df, test_df, product_profiles

def build_profiles_and_features(df):
    # Builds user profiles (positive/negative, preferred categories/prices) and generic fallbacks.
    print("[PHASE 3] Building user profiles and features...")
    
    user_fav_tier = {}
    user_fav_cat = {}
    for user in df['user_id'].unique():
        pos_reviews = df[(df['user_id'] == user) & (df['rating'] >= 4)]
        if not pos_reviews.empty:
            tier_counts = Counter(pos_reviews['price_tier'])
            cat_counts = Counter(pos_reviews['specific_category'])
            if tier_counts: user_fav_tier[user] = tier_counts.most_common(1)[0][0]
            if cat_counts: user_fav_cat[user] = cat_counts.most_common(1)[0][0]

    positive_profiles, negative_profiles = {}, {}
    for user in df['user_id'].unique():
        pos_embs = df[(df['user_id'] == user) & (df['rating'] >= 4)]['review_emb'].tolist()
        neg_embs = df[(df['user_id'] == user) & (df['rating'] <= 2)]['review_emb'].tolist()
        if pos_embs: positive_profiles[user] = np.mean(pos_embs, axis=0)
        if neg_embs: negative_profiles[user] = np.mean(neg_embs, axis=0)

    EMBEDDING_DIM = len(df['review_emb'].iloc[0])
    generic_fallback_emb = np.zeros(EMBEDDING_DIM)

    all_pos_embs = df[df['rating'] >= 4]['review_emb'].tolist()
    generic_pos_profile = np.mean([np.array(e) for e in all_pos_embs], axis=0) if all_pos_embs else generic_fallback_emb

    all_neg_embs = df[df['rating'] <= 2]['review_emb'].tolist()
    generic_neg_profile = np.mean([np.array(e) for e in all_neg_embs], axis=0) if all_neg_embs else generic_fallback_emb

    profiles = {
        "positive": positive_profiles, "negative": negative_profiles,
        "fav_tier": user_fav_tier, "fav_cat": user_fav_cat,
        "generic_pos": generic_pos_profile, "generic_neg": generic_neg_profile,
        "fallback_emb": generic_fallback_emb
    }
    return df, profiles

def prepare_model_data(df, product_profiles, profiles):
    # Prepares the feature vectors (X) and labels (y) for the model.
    x, y = [], []
    EMBEDDING_DIM = len(profiles["fallback_emb"])

    def cosine_similarity(v1, v2):
        if v1 is None or v2 is None or np.all(v1 == 0) or np.all(v2 == 0): return 0
        return 1 - np.clip(cosine(v1, v2), 0.0, 2.0)

    for _, row in df.iterrows():
        user_id, product_id = row['user_id'], row['product_id']

        prod_emb = np.array(product_profiles.get(product_id, profiles["fallback_emb"]))
        review_emb = np.array(row['review_emb'])
        user_pos_emb = np.array(profiles["positive"].get(user_id, profiles["fallback_emb"]))

        pos_profile = profiles["positive"].get(user_id, profiles["generic_pos"])
        neg_profile = profiles["negative"].get(user_id, profiles["generic_neg"])

        sim_pos = cosine_similarity(prod_emb, pos_profile)
        sim_neg = cosine_similarity(prod_emb, neg_profile)

        price_match = 1 if row.get('price_tier') == profiles["fav_tier"].get(user_id) else 0
        cat_match = 1 if row.get('specific_category') == profiles["fav_cat"].get(user_id) else 0

        input_vect = np.concatenate([
            user_pos_emb, prod_emb, review_emb,
            [sim_pos], [sim_neg], [sim_pos - sim_neg], [price_match], [cat_match]
        ])

        x.append(input_vect)
        y.append(1 if row['rating'] >= 4 else 0)

    if not x:
        num_features = (EMBEDDING_DIM * 3) + 5
        return np.empty((0, num_features)), np.empty(0)

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


# ==============================================================================
# 3. MODEL TRAINING AND EVALUATION FUNCTIONS
# ==============================================================================

def train_evaluate_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    # Trains and evaluates an XGBoost classifier.
    print("  - Training XGBoost...")
    count_neg, count_pos = np.sum(y_train == 0), np.sum(y_train == 1)
    scale_pos_weight = (count_neg / count_pos) * 1.5 if count_pos > 0 else 1

    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'eta': 0.05,
        'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.7,
        'scale_pos_weight': scale_pos_weight, 'random_state': 42,
        'gamma': 2, 'lambda': 1.5, 'alpha': 0.5
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    start_time = time.time()
    model = xgb.train(params, dtrain, num_boost_round=2000,
                      evals=[(dtrain, 'train'), (dval, 'eval')], early_stopping_rounds=30, verbose_eval=False)
    training_time = time.time() - start_time
    print(f"  - Training time: {training_time:.2f} secondi")

    start_time_inf = time.time()
    y_pred_proba = model.predict(dtest)
    inference_time = (time.time() - start_time_inf) / len(X_test) if len(X_test) > 0 else 0

    model_filename = "temp_xgb_model.json"
    model.save_model(model_filename)
    model_size_kb = os.path.getsize(model_filename) / 1024
    os.remove(model_filename)
    print(f"  - Final model size: {model_size_kb:.2f} KB")

    y_pred = (y_pred_proba > 0.5).astype(int)

    print("  - XGBoost results:")
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  - AUC sul Test Set: {auc:.4f}")

    return {
        "auc": auc, "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "training_time_sec": training_time,
        "inference_ms_per_sample": inference_time * 1000,
        "model_size_kb": model_size_kb
    }

def create_mlp_model(input_shape):
    # Defines the MLP model architecture.
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(256, activation='relu'), Dropout(0.4),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

def train_evaluate_mlp_centralized(X_train, y_train, X_val, y_val, X_test, y_test):
    # Trains and evaluates an MLP classifier in a centralized manner.
    print("  - Training Centralized MLP...")
    model = create_mlp_model(X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)

    start_time = time.time()
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=100, batch_size=128, callbacks=[early_stopping], verbose=0)
    training_time = time.time() - start_time
    print(f"  - Training time: {training_time:.2f} secondi")

    start_time_inf = time.time()
    y_pred_proba = model.predict(X_test).flatten()
    inference_time = (time.time() - start_time_inf) / len(X_test) if len(X_test) > 0 else 0

    y_pred = (y_pred_proba > 0.5).astype(int)

    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    model_size_kb = (trainable_params * 4) / 1024
    print(f"  - Model size (estimated): {model_size_kb:.2f} KB")

    print("  - Centralized MLP results:")
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  - AUC on Test Set: {auc:.4f}")

    return {
        "auc": auc, "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "training_time_sec": training_time,
        "inference_ms_per_sample": inference_time * 1000,
        "model_size_kb": model_size_kb
    }

# ==============================================================================
# 4. FUNCTION FOR FEDERATED TRAINING
# ==============================================================================
def train_evaluate_federated_mlp(client_features, client_labels, X_val, y_val, X_test, y_test):
    # Trains an MLP with FedAvg, optimized for low RAM consumption.
    print("  - Training Federated MLP...")

    # FL Hyperparameters
    COMMUNICATION_ROUNDS = 30
    CLIENTS_PER_ROUND = 20
    LOCAL_EPOCHS = 1
    LOCAL_BATCH_SIZE = 32

    # Model initialization
    input_shape = client_features[0].shape[1]
    global_model = create_mlp_model(input_shape)

    local_model = create_mlp_model(input_shape)

    num_clients = len(client_features)
    start_time = time.time()

    for round_num in range(COMMUNICATION_ROUNDS):
        client_indices = np.random.choice(num_clients, size=CLIENTS_PER_ROUND, replace=False)

        local_weights_list = []
        client_data_sizes = []

        batch_labels = np.concatenate([client_labels[i] for i in client_indices])
        if len(batch_labels) > 0:
            counts = np.bincount(batch_labels.astype(int), minlength=2)
            print(f"  - Round {round_num + 1}/{COMMUNICATION_ROUNDS} | Client Batch Dist: C0={counts[0]}, C1={counts[1]}", end="")

        for i in client_indices:
            X_local, y_local = client_features[i], client_labels[i]

            local_model.set_weights(global_model.get_weights())

            weights = compute_class_weight('balanced', classes=np.unique(y_local), y=y_local)
            class_weight_dict = dict(zip(np.unique(y_local), weights))

            local_model.fit(X_local, y_local,
                            epochs=LOCAL_EPOCHS,
                            batch_size=LOCAL_BATCH_SIZE,
                            class_weight=class_weight_dict,
                            verbose=0)

            local_weights_list.append(local_model.get_weights())
            client_data_sizes.append(len(X_local))

        total_data_size = sum(client_data_sizes)
        if total_data_size == 0:
            print(" | No data in this round, skipping.")
            continue

        scaling_factors = [size / total_data_size for size in client_data_sizes]

        averaged_weights = []
        for weights_tuple in zip(*local_weights_list):
            layer_mean = tf.math.reduce_sum([w * s for w, s in zip(weights_tuple, scaling_factors)], axis=0)
            averaged_weights.append(layer_mean)

        global_model.set_weights(averaged_weights)

        val_loss, val_auc = global_model.evaluate(X_val, y_val, verbose=0)
        print(f" | Val AUC: {val_auc:.4f}")

        gc.collect()

    training_time = time.time() - start_time
    print(f"  - Total training time (federated): {training_time:.2f} seconds")

    start_time_inf = time.time()
    y_pred_proba = global_model.predict(X_test).flatten()
    inference_time = (time.time() - start_time_inf) / len(X_test) if len(X_test) > 0 else 0
    y_pred = (y_pred_proba > 0.5).astype(int)

    trainable_params = np.sum([np.prod(v.shape) for v in global_model.trainable_weights])
    model_size_kb = (trainable_params * 4) / 1024
    print(f"  - Model size (estimated): {model_size_kb:.2f} KB")

    print("  - Federated MLP Results:")
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  - AUC on Test Set: {auc:.4f}")

    return {
        "auc": auc, "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "training_time_sec": training_time,
        "inference_ms_per_sample": inference_time * 1000,
        "model_size_kb": model_size_kb
    }


# ==============================================================================
# 5. EXPERIMENT
# ==============================================================================

def main():

    all_results = []

    for min_interactions in USER_INTERACTION_THRESHOLDS:
        print(f"\n\n{'#'*60}")
        print(f"### STARTING EXPERIMENT CYCLE WITH USER FILTER >= {min_interactions} ###")
        print(f"{'#'*60}\n")

        # --- PHASE 1: Loading and preprocessing ---
        df_filtered = load_and_preprocess_data(
            "Amazon_Fashion.jsonl.gz", 
            "meta_Amazon_Fashion.jsonl.gz",
            min_user_interactions=min_interactions  
        )

        if df_filtered.empty:
            print(f"No data for threshold {min_interactions}. Skipping to the next.")
            continue

        print("[PHASE 1.5] Creating global features (price_tier, category)...")
        df_filtered['specific_category'] = df_filtered['categories'].apply(lambda cat: cat[-1] if isinstance(cat, list) and cat else 'Unknown')

        # Handle small datasets where the split might fail
        if df_filtered['user_id'].nunique() < 3:
             print(f"Not enough users ({df_filtered['user_id'].nunique()}) to create train/val/test splits. Skipping threshold {min_interactions}.")
             continue

        train_data, val_data, test_data = split_data_by_user(df_filtered)

        try:
            _, price_bins = pd.qcut(train_data['price'].dropna(), q=4, labels=['low', 'medium', 'high', 'luxury'], retbins=True, duplicates='drop')
        except ValueError:
            print("Attenzione: qcut fallito, uso 3 bin invece di 4.")
            _, price_bins = pd.qcut(train_data['price'].dropna(), q=3, labels=['low', 'medium', 'high'], retbins=True, duplicates='drop')

        price_bins[0], price_bins[-1] = -np.inf, np.inf
        labels = ['low', 'medium', 'high', 'luxury'] if len(price_bins) == 5 else ['low', 'medium', 'high']

        for df in [train_data, val_data, test_data]:
            df['price_tier'] = pd.cut(df['price'], bins=price_bins, labels=labels, include_lowest=True)
            df['price_tier'].fillna('medium', inplace=True)

        for model_name in EMBEDDING_MODELS_TO_TEST:
            print(f"\n{'='*25}\nSTARTING EXPERIMENT WITH EMBEDDER: {model_name}\n{'='*25}")

            sbert_model = SentenceTransformer(model_name)
            train_emb, val_emb, test_emb, product_profiles = generate_embeddings(
                sbert_model, model_name, train_data.copy(), val_data.copy(), test_data.copy()
            )
            _, user_profiles = build_profiles_and_features(train_emb)

            print("[PHASE 4] Preparing data for training...")
            client_data_silos = {user_id: group for user_id, group in train_emb.groupby('user_id')}
            train_users = train_emb['user_id'].unique()

            all_client_features, all_client_labels = [], []
            for user_id in train_users:
                client_df = client_data_silos.get(user_id)
                if client_df is None: continue
                x_local, y_local = prepare_model_data(client_df, product_profiles, user_profiles)
                if len(x_local) > 0 and len(np.unique(y_local)) >= 2:
                    all_client_features.append(x_local)
                    all_client_labels.append(y_local)

            print(f"Data prepared for {len(all_client_features)} clients (informative users).")

            if not all_client_features:
                 print("No informative clients for training. Skipping this embedder.")
                 continue

            X_val, y_val = prepare_model_data(val_emb, product_profiles, user_profiles)
            X_test, y_test = prepare_model_data(test_emb, product_profiles, user_profiles)
            X_train_centralized = np.vstack(all_client_features)
            y_train_centralized = np.hstack(all_client_labels)
            print(f"Dimensions: Train (centralized)={X_train_centralized.shape}, Val={X_val.shape}, Test={X_test.shape}")

            for classifier_name in CLASSIFIERS_TO_TEST:
                print(f"\n--- [PHASE 5] Training with CLASSIFIER: {classifier_name.upper()} ---")

                if classifier_name == 'xgboost':
                    metrics = train_evaluate_xgboost(X_train_centralized, y_train_centralized, X_val, y_val, X_test, y_test)
                elif classifier_name == 'mlp_centralized':
                    metrics = train_evaluate_mlp_centralized(X_train_centralized, y_train_centralized, X_val, y_val, X_test, y_test)
                elif classifier_name == 'mlp_federated':
                    metrics = train_evaluate_federated_mlp(all_client_features, all_client_labels, X_val, y_val, X_test, y_test)

                metrics['min_user_interactions'] = min_interactions
                metrics['embedder'] = model_name
                metrics['classifier'] = classifier_name
                all_results.append(metrics)

        print(f"\n--- Clearing memory before the next experiment cycle ---")
        del df_filtered, train_data, val_data, test_data
        gc.collect()

    print("\n\n" + "="*60)
    print("      FINAL SUMMARY OF ALL EXPERIMENTS     ")
    print("="*60)

    if not all_results:
        print("No results to display. Check filters and input data.")
        return

    results_df = pd.DataFrame(all_results)

    final_columns = [
        'min_user_interactions', 'embedder', 'classifier', 'auc', 'f1_macro',
        'training_time_sec', 'inference_ms_per_sample', 'model_size_kb'
    ]

    results_df = results_df.sort_values(by=['min_user_interactions', 'embedder', 'classifier'])
    print(results_df[final_columns].round(4))

if __name__ == "__main__":

    main()


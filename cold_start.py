# ==============================================================================
# TEST COLD START
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


EMBEDDING_MODELS_TO_TEST = [
    'all-mpnet-base-v2',        
    # 'all-MiniLM-L6-v2',         
]

CLASSIFIERS_TO_TEST = [
    'xgboost',
]

CHUNKSIZE = 50000

def load_and_preprocess_data(review_path, meta_path):

    print("\n[FASE 1] Caricamento e preprocessing dei dati (modalità memory-safe)...")

    # --- Carica i metadati ---
    # Il file dei metadati è solitamente più piccolo e può essere caricato interamente.
    print("  - Caricamento metadati dei prodotti...")
    meta = pd.read_json(meta_path, lines=True, compression='gzip')
    meta = meta[['parent_asin', 'title', 'description', 'price', 'categories']]
    meta.columns = ['product_id', 'title', 'description', 'price', 'categories']
    # Rimuoviamo subito i prodotti con titoli troppo corti o mancanti
    meta.dropna(subset=['title'], inplace=True)
    meta = meta[meta['title'].str.len() > 10]

    # --- PRIMA PASSATA: Identificazione di utenti e prodotti attivi ---
    # Leggiamo il file delle recensioni a blocchi solo per contare le occorrenze,
    # senza tenere i dati in memoria.
    print("  - Passata 1: Conteggio recensioni per identificare utenti e prodotti attivi...")
    user_counts = Counter()
    product_counts = Counter()

    review_iterator = pd.read_json(review_path, lines=True, compression='gzip', chunksize=CHUNKSIZE)

    for i, chunk in enumerate(review_iterator):
        print(f"\r    - Conteggio nel blocco {i+1}...", end="")
        user_counts.update(chunk['user_id'])
        product_counts.update(chunk['asin']) # 'asin' è il product_id nel file originale
    print("\n  - Conteggio completato.")

    active_users = {user for user, count in user_counts.items() if count >= 5}
    active_products = {prod for prod, count in product_counts.items() if count >= 3}

    print(f"  - Utenti attivi (>= 5 recensioni): {len(active_users)}")
    print(f"  - Prodotti attivi (>= 3 recensioni): {len(active_products)}")

    del user_counts, product_counts
    gc.collect()

    print("  - Passata 2: Filtraggio e costruzione del DataFrame finale...")
    filtered_chunks = []
    review_iterator = pd.read_json(review_path, lines=True, compression='gzip', chunksize=CHUNKSIZE)

    for i, chunk in enumerate(review_iterator):
        print(f"\r    - Filtraggio del blocco {i+1}...", end="")
        chunk = chunk[['user_id', 'asin', 'text', 'rating']]
        chunk.rename(columns={'asin': 'product_id', 'text': 'review_text'}, inplace=True)

        chunk = chunk[chunk['user_id'].isin(active_users)]
        chunk = chunk[chunk['product_id'].isin(active_products)]

        merged_chunk = pd.merge(chunk, meta, on='product_id', how='inner')

        if not merged_chunk.empty:
            filtered_chunks.append(merged_chunk)

    print("\n  - Concatenazione dei blocchi filtrati...")
    if not filtered_chunks:
        raise ValueError("Nessun dato rimasto dopo il filtraggio. Controlla i percorsi dei file e i criteri di filtro.")

    df_filtered = pd.concat(filtered_chunks, ignore_index=True)

    final_columns = ['user_id', 'product_id', 'review_text', 'rating', 'title', 'description', 'price', 'categories']
    df_filtered = df_filtered[final_columns]

    print(f"  - Processo completato. Dataset finale: {len(df_filtered)} recensioni.")

    del filtered_chunks, meta
    gc.collect()

    return df_filtered

def split_data_by_user(df):
    # Divide il dataset in train, validation e test set mantenendo gli utenti separati.
    print("[FASE 1] Suddivisione degli utenti in train/val/test...")
    all_users = df['user_id'].unique()
    train_val_users, test_users = train_test_split(all_users, test_size=0.15, random_state=42)
    train_users, val_users = train_test_split(train_val_users, test_size=(0.10/0.85), random_state=42)

    train_data = df[df['user_id'].isin(train_users)].copy()
    val_data = df[df['user_id'].isin(val_users)].copy()
    test_data = df[df['user_id'].isin(test_users)].copy()

    print(f"Utenti: {len(train_users)} train, {len(val_users)} validation, {len(test_users)} test.")
    return train_data, val_data, test_data

def generate_embeddings(sbert_model, model_name_str, train_df, val_df, test_df):
    # Calcola gli embedding per recensioni e prodotti.
    print(f"[FASE 2] Calcolo embedding con il modello '{model_name_str}'...")

    for df, name in [(train_df, "Train"), (val_df, "Validation"), (test_df, "Test")]:
      if df.empty:
          print(f"  - Saltando il calcolo embedding per il {name} Set (vuoto).")
          continue

      print(f"  - Calcolo embedding recensioni per il {name} Set...")
      texts = df['review_text'].tolist()
      df['review_emb'] = sbert_model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    print("  - Calcolo embedding prodotti...")
    train_df.loc[:,'description'] = train_df['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    train_df.loc[:,'product_text'] = train_df['title'].fillna('') + ' ' + train_df['description'].fillna('')

    product_texts = train_df.drop_duplicates('product_id')[['product_id', 'product_text']]
    product_embs = sbert_model.encode(product_texts['product_text'].tolist(), batch_size=64, show_progress_bar=True)
    product_profiles = dict(zip(product_texts['product_id'], product_embs.tolist()))

    return train_df, val_df, test_df, product_profiles

def build_profiles_and_features(df):
    # Costruisce i profili utente (positivi/negativi, categorie/prezzi preferiti) e i fallback generici.
    print("[FASE 3] Costruzione profili utente e features...")

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

def prepare_model_data(df, product_profiles, profiles, sbert_model=None):
    # Prepara i vettori di feature (X) e le etichette (y) per il modello.
    x, y = [], []
    EMBEDDING_DIM = len(profiles["fallback_emb"])

    def cosine_similarity(v1, v2):
        if v1 is None or v2 is None or np.all(v1 == 0) or np.all(v2 == 0): return 0
        return 1 - np.clip(cosine(v1, v2), 0.0, 2.0)

    for _, row in df.iterrows():
        user_id, product_id = row['user_id'], row['product_id']

        if product_id in product_profiles:
          # Caso "Warm": l'embedding del prodotto è pre-calcolato
          prod_emb = np.array(product_profiles.get(product_id))
        elif sbert_model is not None:
            # Caso "Cold": l'embedding non esiste, lo calcoliamo al volo
            product_text = str(row['title']) + ' ' + ' '.join(row['description']) if isinstance(row['description'], list) else str(row['description'])
            prod_emb = sbert_model.encode(product_text)
        else:
            prod_emb = np.array(profiles["fallback_emb"])

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

def train_evaluate_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Addestra e valuta un classificatore XGBoost."""
    print("  - Addestramento XGBoost...")
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
    print(f"  - Tempo di addestramento: {training_time:.2f} secondi")

    start_time_inf = time.time()
    y_pred_proba = model.predict(dtest)
    inference_time = (time.time() - start_time_inf) / len(X_test) if len(X_test) > 0 else 0

    model_filename = "temp_xgb_model.json"
    model.save_model(model_filename)
    model_size_kb = os.path.getsize(model_filename) / 1024
    os.remove(model_filename)
    print(f"  - Dimensione modello finale: {model_size_kb:.2f} KB")

    y_pred = (y_pred_proba > 0.5).astype(int)

    print("  - Risultati XGBoost:")
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  - AUC sul Test Set: {auc:.4f}")

    return {
        "auc": auc, "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "training_time_sec": training_time,
        "inference_ms_per_sample": inference_time * 1000,
        "model_size_kb": model_size_kb
    }

# ==============================================================================
# 5. ORCHESTRAZIONE DEGLI ESPERIMENTI
# ==============================================================================

def main():
    """Funzione principale che orchestra l'esperimento di valutazione cold-start."""
    df_filtered = load_and_preprocess_data("Amazon_Fashion.jsonl.gz", "meta_Amazon_Fashion.jsonl.gz")

    df_filtered['specific_category'] = df_filtered['categories'].apply(lambda cat: cat[-1] if isinstance(cat, list) and cat else 'Unknown')

    print("\n[FASE 1.5] Suddivisione dati per scenari Warm, User-Cold, Item-Cold...")

    all_users = df_filtered['user_id'].unique()
    train_val_users, test_users_cold = train_test_split(all_users, test_size=0.15, random_state=42)
    train_users, val_users = train_test_split(train_val_users, test_size=0.15, random_state=42) # 15% del rimanente per validation

    all_products = df_filtered['product_id'].unique()
    train_products, test_products_cold = train_test_split(all_products, test_size=0.15, random_state=42)

    # --- Creazione dei DataFrame ---
    # Set di training: contiene solo utenti e prodotti di training
    train_data = df_filtered[
        df_filtered['user_id'].isin(train_users) &
        df_filtered['product_id'].isin(train_products)
    ]
    # Set di validazione: utenti di validazione, prodotti di training
    val_data = df_filtered[
        df_filtered['user_id'].isin(val_users) &
        df_filtered['product_id'].isin(train_products)
    ]

    # SCENARIO 1: WARM START TEST SET
    # Interazioni di utenti di training su prodotti di training (ma non viste in train_data)
    # Prendiamo un campione casuale dal set di training e lo rimuoviamo
    test_data_warm = train_data.sample(frac=0.1, random_state=42)
    train_data = train_data.drop(test_data_warm.index)

    # SCENARIO 2: USER COLD-START TEST SET
    # Interazioni di utenti NUOVI su prodotti NOTI (di training)
    test_data_user_cold = df_filtered[
        df_filtered['user_id'].isin(test_users_cold) &
        df_filtered['product_id'].isin(train_products)
    ]

    # SCENARIO 3: ITEM COLD-START TEST SET
    # Interazioni di utenti NOTI (di training) su prodotti NUOVI
    test_data_item_cold = df_filtered[
        df_filtered['user_id'].isin(train_users) &
        df_filtered['product_id'].isin(test_products_cold)
    ]

    print(f"  - Dati di Training: {len(train_data)} interazioni")
    print(f"  - Dati di Validazione: {len(val_data)} interazioni")
    print(f"  - Test 'Warm Start': {len(test_data_warm)} interazioni")
    print(f"  - Test 'User Cold-Start': {len(test_data_user_cold)} interazioni")
    print(f"  - Test 'Item Cold-Start': {len(test_data_item_cold)} interazioni")


    try:
        _, price_bins = pd.qcut(train_data['price'], q=4, retbins=True, duplicates='drop')
        labels = ['low', 'medium', 'high', 'luxury']
    except ValueError:
        _, price_bins = pd.qcut(train_data['price'], q=3, retbins=True, duplicates='drop')
        labels = ['low', 'medium', 'high']
    price_bins[0], price_bins[-1] = -np.inf, np.inf

    for df in [train_data, val_data, test_data_warm, test_data_user_cold, test_data_item_cold]:
        df['price_tier'] = pd.cut(df['price'], bins=price_bins, labels=labels, include_lowest=True).astype(str).fillna('medium')

    final_evaluation_results = []

    for model_name in EMBEDDING_MODELS_TO_TEST:
        print(f"\n{'='*25}\nESPERIMENTO CON EMBEDDER: {model_name}\n{'='*25}")
        sbert_model = SentenceTransformer(model_name)

        train_emb, _, _, product_profiles = generate_embeddings(sbert_model, model_name, train_data.copy(), pd.DataFrame(), pd.DataFrame())
        _, user_profiles = build_profiles_and_features(train_emb)

        for df in [val_data, test_data_warm, test_data_user_cold, test_data_item_cold]:
            df['review_emb'] = sbert_model.encode(df['review_text'].tolist(), batch_size=64, show_progress_bar=True).tolist()

        # Fase 4: Preparazione dati per i modelli
        print("[FASE 4] Preparazione dati per l'addestramento e la valutazione...")

        X_val, y_val = prepare_model_data(val_data, product_profiles, user_profiles, sbert_model)

        test_sets = {
            "Warm Start": prepare_model_data(test_data_warm, product_profiles, user_profiles, sbert_model),
            "User Cold-Start": prepare_model_data(test_data_user_cold, product_profiles, user_profiles, sbert_model),
            "Item Cold-Start": prepare_model_data(test_data_item_cold, product_profiles, user_profiles, sbert_model)
        }

        # Dati di training
        client_data_silos = {user_id: group for user_id, group in train_emb.groupby('user_id')}
        train_user_ids = train_emb['user_id'].unique()
        all_client_features, all_client_labels = [], []
        for user_id in train_user_ids:
            client_df = client_data_silos.get(user_id)
            if client_df is None: continue
            x_local, y_local = prepare_model_data(client_df, product_profiles, user_profiles, sbert_model)
            if len(x_local) > 0 and len(np.unique(y_local)) >= 2:
                all_client_features.append(x_local)
                all_client_labels.append(y_local)

        X_train_centralized = np.vstack(all_client_features)
        y_train_centralized = np.hstack(all_client_labels)

        for classifier_name in CLASSIFIERS_TO_TEST:
            print(f"\n--- [FASE 5] Training del classificatore: {classifier_name.upper()} ---")

            if classifier_name == 'xgboost':
                dtrain = xgb.DMatrix(X_train_centralized, label=y_train_centralized)
                dval = xgb.DMatrix(X_val, label=y_val)
                params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'eta': 0.05, 'max_depth': 4}
                trained_model = xgb.train(params, dtrain, num_boost_round=2000, evals=[(dval, 'eval')], early_stopping_rounds=30, verbose_eval=False)

            # --- Valutazione sui 3 scenari ---
            print(f"--- [FASE 6] Valutazione di {classifier_name.upper()} sui diversi scenari ---")
            for scenario_name, (X_test, y_test) in test_sets.items():
                if len(X_test) == 0: continue

                if classifier_name == 'xgboost':
                    dtest = xgb.DMatrix(X_test)
                    y_pred_proba = trained_model.predict(dtest)
                else: # MLP
                    y_pred_proba = trained_model.predict(X_test).flatten()

                y_pred = (y_pred_proba > 0.5).astype(int)

                auc = roc_auc_score(y_test, y_pred_proba)
                f1 = f1_score(y_test, y_pred, average='macro')

                print(f"  - Scenario: {scenario_name:18} | AUC: {auc:.4f} | F1-Macro: {f1:.4f}")

                final_evaluation_results.append({
                    "embedder": model_name,
                    "classifier": classifier_name,
                    "scenario": scenario_name,
                    "auc": auc,
                    "f1_macro": f1
                })

    # ==========================================================================
    # RIEPILOGO FINALE CON TABELLA COMPARATIVA
    # ==========================================================================
    print("\n\n" + "="*70)
    print("      RIEPILOGO FINALE DELLE PERFORMANCE COLD-START     ")
    print("="*70)

    if not final_evaluation_results:
        print("Nessun risultato da visualizzare.")
        return

    results_df = pd.DataFrame(final_evaluation_results)

    auc_pivot = results_df.pivot_table(
        index=['embedder', 'classifier'],
        columns='scenario',
        values='auc'
    )

    f1_pivot = results_df.pivot_table(
        index=['embedder', 'classifier'],
        columns='scenario',
        values='f1_macro'
    )

    print("\n--- Confronto AUC Score ---\n")
    print(auc_pivot.round(4))

    print("\n\n--- Confronto F1-Score (Macro) ---\n")
    print(f1_pivot.round(4))
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

# src/train_autoencoders.py

import os
import re
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import optuna
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import math # For checking nan

def get_column_groups(csv_path):
    """
    Scans the header of a CSV to identify and group embedding columns without loading data.
    """
    print("[REDACTED_BY_SCRIPT]")
    try:
        columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        exit()

    # [FIXED] Corrected patterns to be plural (sps, flaws, features)
    patterns = {
        'sps': r'[REDACTED_BY_SCRIPT]',
        'flaws': r'[REDACTED_BY_SCRIPT]',
        'features': r'[REDACTED_BY_SCRIPT]',
        'persona': r'[REDACTED_BY_SCRIPT]'
    }

    embedding_cols_dict = {}
    all_embedding_cols = set()
    for concept, pattern in patterns.items():
        cols = [col for col in columns if re.search(pattern, col)]
        if cols:
            embedding_cols_dict[concept] = cols
            all_embedding_cols.update(cols)
            print(f"[REDACTED_BY_SCRIPT]'{concept}'.")
    
    if not all_embedding_cols:
        print("[REDACTED_BY_SCRIPT]")
        exit()

    numerical_cols = [col for col in columns if col not in all_embedding_cols]
    print(f"[REDACTED_BY_SCRIPT]")
    
    return numerical_cols, embedding_cols_dict


def preprocess_csv_in_chunks(csv_path, numerical_cols, embedding_cols_dict, chunk_size=1000):
    """
    Reads a large CSV in chunks, processes it, and saves data to intermediate .npy files.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # Initialize lists to hold data from all chunks
    numericals_list = []
    embedding_data_lists = {concept: [] for concept in embedding_cols_dict.keys()}
    
    total_rows = 0
    # Use the more robust 'python' engine for parsing
    with pd.read_csv(csv_path, chunksize=chunk_size, engine='python', on_bad_lines='warn') as reader:
        for i, chunk in enumerate(reader):
            print(f"[REDACTED_BY_SCRIPT]")
            
            # Process numericals
            numericals_list.append(chunk[numerical_cols])

            # Process embeddings for each concept
            for concept, cols in embedding_cols_dict.items():
                embedding_data_lists[concept].append(chunk[cols].astype(np.float32).values)
            
            total_rows += len(chunk)

    print(f"[REDACTED_BY_SCRIPT]")

    # Concatenate and save numericals
    print("[REDACTED_BY_SCRIPT]")
    numericals_df = pd.concat(numericals_list, ignore_index=True)
    numericals_df.to_pickle("numericals.pkl") # Pickle is faster and preserves types

    # Concatenate and save each embedding group as a .npy file
    for concept, data_list in embedding_data_lists.items():
        print(f"[REDACTED_BY_SCRIPT]'{concept}'...")
        full_array = np.vstack(data_list)
        np.save(f"[REDACTED_BY_SCRIPT]", full_array)
        print(f"[REDACTED_BY_SCRIPT]")
    
    print("[REDACTED_BY_SCRIPT]")
    return list(embedding_cols_dict.keys())


def build_autoencoder(input_dim, latent_dim, layer_1_size, layer_2_size):
    """
    Defines and compiles the Keras autoencoder model architecture with dynamic layer sizes.
    """
    # Encoder
    encoder_input = keras.layers.Input(shape=(input_dim,), name='encoder_input')
    x = keras.layers.Dense(layer_1_size, activation='relu')(encoder_input)
    x = keras.layers.Dense(layer_2_size, activation='relu')(x)
    bottleneck = keras.layers.Dense(latent_dim, activation='linear', name='bottleneck')(x)
    encoder = keras.Model(encoder_input, bottleneck, name='encoder')

    # Decoder
    decoder_input = keras.layers.Input(shape=(latent_dim,), name='decoder_input')
    x = keras.layers.Dense(layer_2_size, activation='relu')(decoder_input)
    x = keras.layers.Dense(layer_1_size, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dim, activation='linear')(x)
    decoder = keras.Model(decoder_input, decoder_output, name='decoder')
    
    autoencoder = keras.Model(encoder_input, decoder(encoder(encoder_input)), name='autoencoder')
    
    return autoencoder, encoder

def objective(trial, X_train, X_val, input_dim):
    """
    The objective function for Optuna to optimize.
    """
    # 1. Suggest hyperparameters for this trial
    latent_dim = trial.suggest_categorical('latent_dim', [16, 32, 64])
    layer_1_size = trial.suggest_categorical('layer_1_size', [128, 256, 512])
    # Ensure layer 2 is smaller than or equal to layer 1 for a bottleneck effect
    layer_2_size = trial.suggest_categorical('layer_2_size', [64, 128])
    if layer_2_size >= layer_1_size:
        # Pruning tells Optuna this trial is unpromising and should be stopped early.
        raise optuna.exceptions.TrialPruned()
        
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    
    # 2. Build the model with the suggested parameters
    autoencoder, _ = build_autoencoder(input_dim, latent_dim, layer_1_size, layer_2_size)
    
    # Compile with the suggested learning rate AND gradient clipping
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0) # <-- ADD clipnorm=1.0
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Use EarlyStopping to prevent wasting time on bad trials
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 3. Train the model
    history = autoencoder.fit(
        X_train, X_train,
        epochs=30,  # Use a moderate number of epochs for tuning
        batch_size=256,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping],
        verbose=0  # Keep the log clean during tuning
    )
    
    # 4. Return the score that Optuna should minimize
    validation_loss = min(history.history['val_loss'])

    # [FIX] Check for NaN and return a massive number to guide Optuna away.
    if math.isnan(validation_loss):
        return 1e9 # A very large number indicating failure
        
    return validation_loss

def main(args):
    """
    Main execution block with Optuna tuning and final model retraining.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    os.makedirs(args.model_dir, exist_ok=True)
    output_parent_dir = os.path.dirname(args.output_csv)
    if output_parent_dir:
        os.makedirs(output_parent_dir, exist_ok=True)

    # STAGE 1: Pre-process the large CSV into manageable .npy files (same as before)
    numerical_cols, embedding_cols_dict = get_column_groups(args.input_csv)
    concepts_to_process = preprocess_csv_in_chunks(
        args.input_csv, 
        numerical_cols, 
        embedding_cols_dict
    )
    
    # STAGE 2: Find the best hyperparameters for each concept using Optuna
    print("[REDACTED_BY_SCRIPT]")
    best_params_per_concept = {}

    for concept_name in concepts_to_process:
        print(f"[REDACTED_BY_SCRIPT]'{concept_name}' ---")
        embedding_data = np.load(f"[REDACTED_BY_SCRIPT]")
        
        # [NEW] Normalize the data before training
        print(f"[REDACTED_BY_SCRIPT]'{concept_name}'...")
        scaler = StandardScaler()
        embedding_data_scaled = scaler.fit_transform(embedding_data)

        # [NEW] Sanitize the data to remove any NaN or Inf values.
        if np.isnan(embedding_data_scaled).any() or np.isinf(embedding_data_scaled).any():
            print(f"[REDACTED_BY_SCRIPT]'{concept_name}' data. Cleaning...")
            embedding_data_scaled = np.nan_to_num(embedding_data_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Save the scaler so we can use it again later
        import pickle
        with open(f"[REDACTED_BY_SCRIPT]", 'wb') as f:
            pickle.dump(scaler, f)

        # Split the SCALED data for tuning
        X_train, X_val = train_test_split(embedding_data_scaled, test_size=0.2, random_state=42)
        
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        
        study.optimize(
            lambda trial: objective(trial, X_train, X_val, embedding_data.shape[1]), 
            n_trials=args.n_trials,
            n_jobs=-1 # Use all available CPU cores for parallel trials
        )
        
        best_params = study.best_trial.params
        best_params_per_concept[concept_name] = best_params
        print(f"  -> Best trial for '{concept_name}'[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")

    # STAGE 3: Retrain final models using the best parameters and generate features
    print("[REDACTED_BY_SCRIPT]")
    compressed_arrays = {}

    for concept_name, best_params in best_params_per_concept.items():
        print(f"[REDACTED_BY_SCRIPT]'{concept_name}' ---")
        
        # Load the full dataset for this concept
        embedding_data = np.load(f"[REDACTED_BY_SCRIPT]")
    
        # [NEW] Load the saved scaler and transform the data
        import pickle
        with open(f"[REDACTED_BY_SCRIPT]", 'rb') as f:
            scaler = pickle.load(f)
        embedding_data_scaled = scaler.transform(embedding_data)
        
        # 1. Build the final model with the optimal parameters
        autoencoder, encoder = build_autoencoder(
            input_dim=embedding_data_scaled.shape[1],
            latent_dim=best_params['latent_dim'],
            layer_1_size=best_params['layer_1_size'],
            layer_2_size=best_params['layer_2_size']
        )
        
        # 2. Compile it with the optimal learning rate
        optimizer = Adam(learning_rate=best_params['learning_rate'])
        autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # 3. Train on the ENTIRE dataset for more epochs to get the best possible model
        print("[REDACTED_BY_SCRIPT]")
        autoencoder.fit(
            embedding_data_scaled, embedding_data_scaled,
            epochs=args.epochs, # Use the full number of epochs passed as an argument
            batch_size=256,
            shuffle=True,
            verbose=1
        )
        
        # 4. Save the final, optimized model
        save_path = os.path.join(args.model_dir, f"[REDACTED_BY_SCRIPT]")
        autoencoder.save(save_path)
        print(f"[REDACTED_BY_SCRIPT]'{concept_name}'[REDACTED_BY_SCRIPT]")
        
        # 5. Generate latent features for the final output
        print(f"[REDACTED_BY_SCRIPT]'{concept_name}'...")
        compressed_features = encoder.predict(embedding_data_scaled, batch_size=args.batch_size)
        compressed_arrays[concept_name] = compressed_features

    # STAGE 4: Final Assembly (same as before)
    print("[REDACTED_BY_SCRIPT]")
    numericals_df = pd.read_pickle("numericals.pkl")
    
    compressed_dfs = []
    for concept_name, data_array in compressed_arrays.items():
        latent_df = pd.DataFrame(
            data_array,
            columns=[f'[REDACTED_BY_SCRIPT]' for i in range(data_array.shape[1])],
            index=numericals_df.index
        )
        compressed_dfs.append(latent_df)

    final_df = pd.concat([numericals_df] + compressed_dfs, axis=1)
    
    final_df.to_csv(args.output_csv, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[REDACTED_BY_SCRIPT]")
    parser.add_argument('--input-csv', type=str, required=True, help="[REDACTED_BY_SCRIPT]")
    parser.add_argument('--output-csv', type=str, required=True, help="[REDACTED_BY_SCRIPT]")
    parser.add_argument('--model-dir', type=str, required=True, help="[REDACTED_BY_SCRIPT]")
    # You can keep the --latent-dim argument as a fallback or remove it, as Optuna now controls it.
    # It's no longer used by the main training loop.
    parser.add_argument('--epochs', type=int, default=75, help="[REDACTED_BY_SCRIPT]")
    parser.add_argument('--batch-size', type=int, default=256, help="[REDACTED_BY_SCRIPT]")
    # New argument for Optuna
    parser.add_argument('--n-trials', type=int, default=25, help="[REDACTED_BY_SCRIPT]")
    
    args = parser.parse_args()
    main(args)
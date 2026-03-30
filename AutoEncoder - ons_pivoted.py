import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l1, l2, l1_l2
import matplotlib.pyplot as plt

# --- Configuration ---
FILE_PATH = r'[REDACTED_BY_SCRIPT]'  # <--- REPLACE WITH YOUR ACTUAL CSV FILE PATH
ID_COLUMN = 'OA21_Code'
EMBEDDING_DIM = 32  # Desired dimensionality of the embedding (e.g., 16, 32, 64)
EPOCHS = 200
BATCH_SIZE = 36

# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]")
    # Create a dummy CSV for demonstration if the file is not found
    print("[REDACTED_BY_SCRIPT]")
    data_for_dummy = {
        'OA21_Code': [f'E{i:08d}' for i in range(100)],
    }
    # Add 120 feature columns with random integers
    for i in range(1, 121):
        data_for_dummy[f'feature_{i}'] = np.random.randint(0, 100, 100)
    df = pd.DataFrame(data_for_dummy)
    FILE_PATH = 'dummy_data.csv'
    df.to_csv(FILE_PATH, index=False)
    print(f"[REDACTED_BY_SCRIPT]")


# Separate identifiers and features
if ID_COLUMN in df.columns:
    identifiers = df[ID_COLUMN]
    features_df = df.drop(columns=[ID_COLUMN])
else:
    print(f"Warning: ID column '{ID_COLUMN}'[REDACTED_BY_SCRIPT]")
    identifiers = pd.Series(range(len(df)), name="generated_id") # Generate dummy IDs
    features_df = df.copy()

print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")

# Convert all feature columns to numeric, coercing errors (e.g., non-numeric strings become NaN)
for col in features_df.columns:
    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

# Handle potential NaNs (e.g., fill with 0 or mean, or drop rows/cols)
# For simplicity, we'll fill with 0. Consider a more sophisticated strategy for real data.
if features_df.isnull().values.any():
    print("[REDACTED_BY_SCRIPT]")
    features_df = features_df.fillna(0)

# Normalize data to be between 0 and 1 (good for neural networks)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features_df)

input_dim = X_scaled.shape[1]

# --- 2. Build the Autoencoder Model ---

# Encoder
input_layer = Input(shape=(input_dim,), name='encoder_input')
encoded = Dense(128, activation='relu', kernel_regularizer=l2(0.002))(input_layer)
encoded = Dropout(0.2)(encoded)
encoded = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(encoded)
encoded = Dense(EMBEDDING_DIM, activation='relu', name='embedding_layer')(encoded) # Bottleneck - our embedding

# Decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dropout(0.2)(decoded)
decoded = Dense(input_dim, activation='sigmoid', name='decoder_output')(decoded) # Output layer activation 'sigmoid' because input was scaled to 0-1

# Autoencoder model (ties encoder and decoder together)
autoencoder = Model(inputs=input_layer, outputs=decoded, name='autoencoder')

# Encoder model (for getting the embeddings later)
encoder_model = Model(inputs=input_layer, outputs=encoded, name='encoder')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.000025, verbose=1, mode='auto', cooldown=0)

# --- 3. Compile the Autoencoder ---
autoencoder.compile(optimizer=Adam(learning_rate=0.0015), loss='mse') # Mean Squared Error for reconstruction

autoencoder.summary()
print("[REDACTED_BY_SCRIPT]")
encoder_model.summary()

# --- 4. Train the Autoencoder ---
print(f"[REDACTED_BY_SCRIPT]")
history = autoencoder.fit(
    X_scaled,          # Input data
    X_scaled,          # Target data (autoencoder tries to reconstruct its input)
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.1, # Use 10% of data for validation
    verbose=1,
    callbacks=[reduce_lr]  # Reduce learning rate on plateau
)

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# --- 5. Generate Embeddings ---
print("[REDACTED_BY_SCRIPT]")
embeddings = encoder_model.predict(X_scaled)

print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")

# --- 6. Save or Use Embeddings ---
# Create a DataFrame for the embeddings with the original identifiers
embedding_columns = [f'embed_{i}' for i in range(EMBEDDING_DIM)]
embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns)

# If you want to add the original identifiers back:
if identifiers is not None:
    embeddings_df = pd.concat([identifiers.reset_index(drop=True), embeddings_df], axis=1)

# Save embeddings to a new CSV file
output_embeddings_file = 'embeddings.csv'
embeddings_df.to_csv(output_embeddings_file, index=False)
print(f"[REDACTED_BY_SCRIPT]")
print("[REDACTED_BY_SCRIPT]")
print(embeddings_df.head())

# --- Optional: Evaluate Reconstruction Quality (Visual Check) ---
# Get reconstructed data
reconstructed_X = autoencoder.predict(X_scaled)

# Compare a sample from original scaled data and reconstructed data
sample_index = 0
print(f"[REDACTED_BY_SCRIPT]")
print("Original (scaled):")
print(X_scaled[sample_index][:10]) # Print first 10 features for brevity
print("Reconstructed:")
print(reconstructed_X[sample_index][:10]) # Print first 10 features

mse_sample = np.mean((X_scaled[sample_index] - reconstructed_X[sample_index])**2)
print(f"[REDACTED_BY_SCRIPT]")

overall_mse = np.mean((X_scaled - reconstructed_X)**2)
print(f"[REDACTED_BY_SCRIPT]")
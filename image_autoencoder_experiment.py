import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

# --- Configuration ---
# UPDATE THIS PATH to your main image directory
MAIN_IMAGE_DIR = r'[REDACTED_BY_SCRIPT]' 
OUTPUT_DIR = 'autoencoder_output'
TARGET_SIZE = (128, 128)  # Target size for all images
LATENT_DIM = 256          # Size of the compressed vector. CRITICAL hyperparameter.
PROPERTY_LIMIT = 100      # Limit the number of properties to process for this PoC
EPOCHS = 30               # Number of training cycles
BATCH_SIZE = 32           # Number of images to process at once

def find_latest_year_for_main(prop_image_base_dir):
    """[REDACTED_BY_SCRIPT]"""
    try:
        year_folders = [d for d in os.listdir(prop_image_base_dir) if os.path.isdir(os.path.join(prop_image_base_dir, d)) and d.isdigit()]
        if not year_folders:
            return None
        return max(year_folders)
    except FileNotFoundError:
        return None

def load_images_from_disk():
    """[REDACTED_BY_SCRIPT]"""
    all_images = []
    print(f"[REDACTED_BY_SCRIPT]")
    
    if not os.path.exists(MAIN_IMAGE_DIR):
        print(f"[REDACTED_BY_SCRIPT]'{MAIN_IMAGE_DIR}'")
        sys.exit(1)
        
    address_folders = [f for f in os.listdir(MAIN_IMAGE_DIR) if os.path.isdir(os.path.join(MAIN_IMAGE_DIR, f))]
    print(f"[REDACTED_BY_SCRIPT]")

    processed_count = 0
    for property_address_folder_name in address_folders:
        if processed_count >= PROPERTY_LIMIT:
            break
        
        prop_image_base_dir = os.path.join(MAIN_IMAGE_DIR, property_address_folder_name)
        latest_year = find_latest_year_for_main(prop_image_base_dir)

        if latest_year:
            year_path = os.path.join(prop_image_base_dir, latest_year)
            image_files = [f for f in os.listdir(year_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                continue

            print(f"[REDACTED_BY_SCRIPT]")
            for image_name in image_files:
                try:
                    image_path = os.path.join(year_path, image_name)
                    with Image.open(image_path) as img:
                        img = img.convert('RGB') # Ensure 3 channels
                        img = img.resize(TARGET_SIZE)
                        img_array = np.array(img) / 255.0 # Normalize to [0, 1]
                        all_images.append(img_array)
                except Exception as e:
                    print(f"[REDACTED_BY_SCRIPT]")
            processed_count += 1

    return np.array(all_images)

def build_autoencoder(input_shape, latent_dim):
    """[REDACTED_BY_SCRIPT]"""
    # --- Encoder ---
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    latent_vector = layers.Dense(latent_dim, name='latent_vector')(x)
    
    encoder = models.Model(encoder_input, latent_vector, name='encoder')

    # --- Decoder ---
    decoder_input = layers.Input(shape=(latent_dim,))
    # Reshape latent vector back into a shape suitable for transposed convolution
    x = layers.Dense(16 * 16 * 128, activation='relu')(decoder_input)
    x = layers.Reshape((16, 16, 128))(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoder_output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = models.Model(decoder_input, decoder_output, name='decoder')

    # --- Full Autoencoder ---
    autoencoder_output = decoder(encoder(encoder_input))
    autoencoder = models.Model(encoder_input, autoencoder_output, name='autoencoder')
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder, decoder

def save_reconstructions(model, test_images, n=10):
    """[REDACTED_BY_SCRIPT]"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    reconstructed_images = model.predict(test_images)

    for i in range(n):
        # Get original and reconstructed images
        original_img_array = (test_images[i] * 255).astype(np.uint8)
        reconstructed_img_array = (reconstructed_images[i] * 255).astype(np.uint8)

        # Convert to PIL Image
        original_pil = Image.fromarray(original_img_array)
        reconstructed_pil = Image.fromarray(reconstructed_img_array)

        # Save images
        original_pil.save(os.path.join(OUTPUT_DIR, f'original_{i}.png'))
        reconstructed_pil.save(os.path.join(OUTPUT_DIR, f'[REDACTED_BY_SCRIPT]'))
        
    print(f"[REDACTED_BY_SCRIPT]'{OUTPUT_DIR}' folder.")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    images = load_images_from_disk()
    if images.shape[0] == 0:
        print("[REDACTED_BY_SCRIPT]")
    else:
        print(f"[REDACTED_BY_SCRIPT]")
        
        # Split data into training and testing sets
        x_train, x_test = train_test_split(images, test_size=0.2, random_state=42)
        print(f"[REDACTED_BY_SCRIPT]")

        # 2. Build Model
        autoencoder, encoder, decoder = build_autoencoder(x_train.shape[1:], LATENT_DIM)
        autoencoder.summary()

        # 3. Train Model
        print("[REDACTED_BY_SCRIPT]")
        autoencoder.fit(x_train, x_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        validation_data=(x_test, x_test))
        
        # 4. Save Results
        save_reconstructions(autoencoder, x_test, n=min(10, len(x_test)))
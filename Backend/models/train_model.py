import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import matplotlib.pyplot as plt
from datetime import datetime

# ========== CONFIGURATION ==========
SOIL_CLASSES = ['chalky', 'clay', 'loamy', 'peaty', 'sandy', 'silty']
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# ✅ FIXED PATHS
DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset", "soil_images")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "soil_classifier.h5")

# ========== STEP 1: LOAD IMAGES ==========
def load_images_from_folder(folder_path):
    images = []
    labels = []
    
    print("\n📂 LOADING IMAGES FROM DATASET...")
    print("-" * 60)
    total_images = 0
    
    for class_idx, soil_type in enumerate(SOIL_CLASSES):
        class_path = os.path.join(folder_path, soil_type)
        
        if not os.path.exists(class_path):
            print(f"⚠️  WARNING: Folder not found: {class_path}")
            continue
        
        # Support common image extensions
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(class_path, ext)))
        
        print(f"📷 {soil_type.upper():10s}: {len(files):4d} images")
        total_images += len(files)
        
        for idx, file in enumerate(files):
            try:
                img = Image.open(file).convert('RGB')
                img = img.resize(IMAGE_SIZE)
                
                # ✅ FIX: Do NOT divide by 255 here. 
                # The Model's Rescaling layer expects 0-255 input.
                img_array = np.array(img) 
                
                images.append(img_array)
                labels.append(class_idx)
                
                if (idx + 1) % 100 == 0:
                    print(f"      Processing {idx + 1}/{len(files)}...", end='\r')
                    
            except Exception as e:
                print(f"⚠️  Error loading {file}: {e}")
        
        if len(files) > 0:
            print(f"      ✅ Loaded {len(files)} images")
    
    print("-" * 60)
    print(f"✅ TOTAL IMAGES LOADED: {total_images}\n")
    
    if len(images) == 0:
        print("❌ ERROR: No images found!")
        print(f"   Dataset path: {DATA_DIR}")
        print(f"   Expected structure:")
        for soil in SOIL_CLASSES:
            print(f"      {DATA_DIR}/{soil}/")
        sys.exit(1)
    
    return np.array(images), np.array(labels)

# ========== STEP 2: BUILD MODEL ==========
def build_model():
    print("\n🔨 BUILDING MODEL ARCHITECTURE...")
    print("-" * 60)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        # ✅ This layer converts 0-255 input to -1 to 1 (Required for MobileNetV2)
        layers.Rescaling(1./127.5, offset=-1), 
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(len(SOIL_CLASSES), activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model built successfully!")
    print(f"   Total parameters: {model.count_params():,}")
    print("-" * 60 + "\n")
    
    return model

# ========== STEP 3: PREPARE DATA ==========
def prepare_data(X, y):
    print("📊 PREPARING DATA...")
    print("-" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=y
    )
    
    print(f"Training set:   {len(X_train)} images ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set:       {len(X_test)} images ({len(X_test)/len(X)*100:.1f}%)")
    
    print("\nClass distribution in training set:")
    for soil_idx, soil_type in enumerate(SOIL_CLASSES):
        count = np.sum(y_train == soil_idx)
        percentage = count / len(y_train) * 100
        print(f"  {soil_type.upper():10s}: {count:4d} ({percentage:5.1f}%)")
    
    print("-" * 60 + "\n")
    
    return X_train, X_test, y_train, y_test

# ========== STEP 4: TRAIN MODEL ==========
def train_model(model, X_train, X_test, y_train, y_test):
    print("🚀 STARTING TRAINING...")
    print("-" * 60)
    print(f"Epochs:        {EPOCHS}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Learning rate: 0.001 (will be reduced if no improvement)")
    print("-" * 60 + "\n")
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    return history

# ========== STEP 5: EVALUATE MODEL ==========
def evaluate_model(model, X_test, y_test):
    print("\n" + "=" * 60)
    print("📈 EVALUATING MODEL...")
    print("=" * 60)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📊 Overall Test Results:")
    print(f"   Loss:     {loss:.4f}")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    
    predictions = model.predict(X_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    print(f"\n📋 Per-Class Accuracy:")
    print("-" * 60)
    for soil_idx, soil_type in enumerate(SOIL_CLASSES):
        mask = y_test == soil_idx
        if np.sum(mask) > 0:
            class_accuracy = np.sum(predicted_labels[mask] == y_test[mask]) / np.sum(mask)
            count = np.sum(mask)
            print(f"   {soil_type.upper():10s}: {class_accuracy * 100:6.2f}% ({count} samples)")
    
    print("-" * 60 + "\n")

# ========== STEP 6: PLOT TRAINING HISTORY ==========
def plot_history(history):
    print("📊 PLOTTING TRAINING HISTORY...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'training_history.png')
    plt.savefig(plot_path, dpi=100)
    print(f"✅ Saved training history plot to: {plot_path}")
    plt.close()

# ========== MAIN EXECUTION BLOCK ==========
def main():
    print("=" * 60)
    print("🌱 SOIL CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Dataset path: {DATA_DIR}")
    print(f"💾 Model will be saved to: {MODEL_PATH}")
    print("=" * 60)

    # 1. Load Data
    X, y = load_images_from_folder(DATA_DIR)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # 3. Build Model
    model = build_model()
    
    # 4. Train
    history = train_model(model, X_train, X_test, y_train, y_test)
    
    # 5. Evaluate
    evaluate_model(model, X_test, y_test)
    
    # 6. Plot
    plot_history(history)
    
    # 7. Save Model
    model.save(MODEL_PATH)
    print(f"💾 Model saved successfully to: {MODEL_PATH}")
    print("=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        sys.exit(1)
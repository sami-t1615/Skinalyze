print("Skinalyze training started (tuned version)...")

import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

# =========================================================
# CONFIG
# =========================================================
IMG_SIZE = 224
BATCH_SIZE = 16

# train a bit longer
EPOCHS_HEAD = 20      # Phase 1
EPOCHS_FINE = 25      # Phase 2

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "dataset", "all")

print("Base dir :", base_dir)
print("Data dir :", data_dir)

# =========================================================
# DATA GENERATORS (slightly softer augmentation)
# =========================================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,          # was 20
    zoom_range=0.20,            # was 0.25
    width_shift_range=0.10,     # was 0.15
    height_shift_range=0.10,    # was 0.15
    shear_range=0.10,           # was 0.15
    brightness_range=[0.85, 1.2],
    horizontal_flip=True,
    validation_split=0.15       # a bit more data for training (was 0.2)
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.15
)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = val_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = train_gen.num_classes
print("Number of classes detected:", num_classes)
print("Class indices:", train_gen.class_indices)

# =========================================================
# MODEL: MobileNetV2 + custom head
# =========================================================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(
    256,
    activation="relu",
    kernel_regularizer=regularizers.l2(1e-4)
)(x)
x = Dropout(0.5)(x)      # a bit more dropout to avoid overfitting
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.summary()

# Callbacks – allow a bit more patience now
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

# =========================================================
# PHASE 1 — TRAIN ONLY THE CLASSIFIER HEAD
# =========================================================
print("\n========== PHASE 1: Training classifier head ==========\n")

base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=[early_stop, reduce_lr]
)

# =========================================================
# PHASE 2 — FINE-TUNE LAST 60 LAYERS OF MOBILENETV2
# =========================================================
print("\n========== PHASE 2: Fine-tuning last 60 layers ==========\n")

base_model.trainable = True

# Freeze all layers except last 60 (deeper fine-tuning)
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=[early_stop, reduce_lr]
)

# =========================================================
# SAVE MODEL & CLASS INDICES
# =========================================================
model_path = os.path.join(base_dir, "skin_mobilenetv2.h5")
model.save(model_path)

class_indices_path = os.path.join(base_dir, "class_indices.json")
with open(class_indices_path, "w") as f:
    json.dump(train_gen.class_indices, f)

print("\nTraining complete.")
print("Model saved to:", model_path)
print("Class indices saved to:", class_indices_path)
print("Skinalyze training finished (tuned version).")

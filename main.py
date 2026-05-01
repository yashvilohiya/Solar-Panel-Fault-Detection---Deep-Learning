import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# Define paths
data_dir = 'Faulty_solar_panel'
img_height, img_width = 224, 224
batch_size = 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% for validation
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load training data
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation data
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Since no separate test set, use validation as test for evaluation
test_generator = validation_generator

# Print class indices
print("Class indices:", train_generator.class_indices)

# ==================== DATASET STATISTICS ====================
print("\n" + "="*70)
print("DATASET STATISTICS")
print("="*70)

# Count images per class
class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
train_counts = np.bincount(train_generator.classes)
val_counts = np.bincount(validation_generator.classes)
total_counts = train_counts + val_counts

# Calculate percentages and imbalance ratio
total_images = len(train_generator.classes) + len(validation_generator.classes)
percentages = (total_counts / total_images) * 100
max_count = np.max(total_counts)
min_count = np.min(total_counts)
imbalance_ratio = max_count / min_count

# Print class distribution table
print("\n{:<20} {:>10} {:>10}".format("Class", "Count", "Share"))
print("-" * 40)
for class_name, count, pct in zip(class_names, total_counts, percentages):
    print("{:<20} {:>10} {:>9.1f}%".format(class_name, count, pct))
print("=" * 40)
print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}x")
print("→ Class weighting will be used to compensate.")

# Print split information
print("\n✓ Split complete.")
print(f"  train : {len(train_generator.classes)} images")
print(f"  val   : {len(validation_generator.classes)} images")
print(f"  test  : {len(validation_generator.classes)} images")

print("\nFound {} images belonging to {} classes.".format(len(train_generator.classes), len(class_names)))
print("Found {} images belonging to {} classes.".format(len(validation_generator.classes), len(class_names)))
print("Found {} images belonging to {} classes.".format(len(validation_generator.classes), len(class_names)))

print("\n📊 Dataset splits:")
print(f"  Train : {len(train_generator.classes)} images")
print(f"  Val   : {len(validation_generator.classes)} images")
print(f"  Test  : {len(validation_generator.classes)} images")

# ==================== VISUALIZATION 1: CLASS DISTRIBUTION ====================
class_labels = train_generator.class_indices
class_counts = np.bincount(train_generator.classes)

plt.figure(figsize=(10, 6))
bars = plt.bar(class_labels.keys(), class_counts)
plt.xlabel('Classes', fontsize=12, fontweight='bold')
plt.ylabel('Number of images', fontsize=12, fontweight='bold')
plt.title('Class Distribution (Training Set)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Class distribution plot saved as 'class_distribution.png'")

# ==================== VISUALIZATION 2: SAMPLE AUGMENTED IMAGES ====================
# Get sample images from each class
augmented_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

fig, axes = plt.subplots(1, len(class_labels), figsize=(16, 3))
fig.suptitle('Sample Augmented Training Images (one per class)', fontsize=14, fontweight='bold')

simple_loader = ImageDataGenerator()

for idx, (class_name, class_idx) in enumerate(class_labels.items()):
    class_dir = os.path.join(data_dir, class_name)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f != 'desktop.ini']
    
    if images:
        img_path = os.path.join(class_dir, images[0])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply simple augmentation for visualization
        aug_iter = simple_loader.flow(img_array, batch_size=1)
        augmented_img = next(aug_iter)[0]
        
        axes[idx].imshow(np.clip(augmented_img, 0, 1))
        axes[idx].set_title(class_name, fontsize=10, fontweight='bold')
        axes[idx].axis('off')

plt.tight_layout()
plt.savefig('augmented_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Augmented samples plot saved as 'augmented_samples.png'")


# Compute class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

print("\n" + "="*52)
print("Class weights (inverse-frequency balanced):")
print("=" * 52)
for idx, class_name in enumerate(class_names):
    weight = class_weights[idx]
    print(f"  {class_name:<25}: {weight:.4f}")
print("=" * 52)
print("  → Higher weight = rarer class = penalized more when misclassified")

# Load pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pool spatial dimensions
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)
model.summary()
print("="*70)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.4f}')

# Predictions
test_generator.reset()
predictions = model.predict(test_generator, verbose=0)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Print classification report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)

# Get per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)))
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

print("\n{:<20} {:>12} {:>12} {:>12} {:>12}".format("CLASS", "PRECISION", "RECALL", "F1", "SUPPORT"))
print("-" * 80)

# Identify classes with F1 < 0.80 (needs attention)
attention_needed = []
for idx, class_name in enumerate(class_names):
    f1_score = f1[idx]
    marker = " ⚠" if f1_score < 0.80 else ""
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>12}{}".format(
        class_name, precision[idx], recall[idx], f1_score, int(support[idx]), marker
    ))
    if f1_score < 0.80:
        attention_needed.append((class_name, f1_score))

print("-" * 80)
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Macro Average", macro_precision, macro_recall, macro_f1))
print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format("Weighted Avg", weighted_precision, weighted_recall, weighted_f1))
print("{:<20} {:>12.4f}".format("Overall Accuracy", test_acc))
print("="*80)

if attention_needed:
    print("\n⚠ Classes with F1 < 0.80 (requires attention):")
    for class_name, f1_score in attention_needed:
        print(f"  - {class_name}: {f1_score:.4f}")
else:
    print("\n✓ All classes have F1 >= 0.80")

# ==================== VISUALIZATION 3: CONFUSION MATRIX ====================
cm = confusion_matrix(y_true, y_pred)
class_names = list(train_generator.class_indices.keys())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
            yticklabels=class_names, ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('Predicted Label', fontsize=11)

# Normalized (Row = Recall per Class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', xticklabels=class_names, 
            yticklabels=class_names, ax=axes[1], cbar_kws={'label': 'Recall per Class'})
axes[1].set_title('Confusion Matrix (Normalized - Row = Recall per Class)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Confusion matrix plot saved as 'confusion_matrix.png'")

# ==================== VISUALIZATION 4: TRAINING HISTORY ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training History', fontsize=14, fontweight='bold')

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
axes[0, 0].set_title('Accuracy', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Val', linewidth=2)
axes[0, 1].set_title('Loss', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision & Recall
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
axes[1, 0].bar(['Precision', 'Recall', 'F1-Score'], [precision, recall, f1], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1, 0].set_title('Precision, Recall, F1-Score (Weighted)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_ylim([0, 1])
for i, v in enumerate([precision, recall, f1]):
    axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Per-class metrics
precision_pc, recall_pc, f1_pc, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
x = np.arange(len(class_names))
width = 0.25
axes[1, 1].bar(x - width, precision_pc, width, label='Precision', alpha=0.8)
axes[1, 1].bar(x, recall_pc, width, label='Recall', alpha=0.8)
axes[1, 1].bar(x + width, f1_pc, width, label='F1-Score', alpha=0.8)
axes[1, 1].set_title('Per-Class Metrics', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Classes')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Training history plot saved as 'training_history.png'")

# ==================== VISUALIZATION 5: SAMPLE PREDICTIONS ====================
predictions = model.predict(test_generator, verbose=0)
y_pred_probs = predictions
y_pred = np.argmax(predictions, axis=1)

# Find correct and incorrect predictions
correct_mask = y_pred == y_true
incorrect_indices = np.where(~correct_mask)[0]
correct_indices = np.where(correct_mask)[0]

# Display mix of correct and incorrect predictions
n_cols = 6
n_rows = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
fig.suptitle('Sample Test Predictions (Green=Correct, Red=Wrong)', fontsize=14, fontweight='bold')

plot_idx = 0
# Create a helper to load original test images for display
test_image_paths = [os.path.join(data_dir, fname) for fname in test_generator.filenames]

# Add some correct predictions
for idx in correct_indices[:n_cols * n_rows // 2]:
    if plot_idx >= n_cols * n_rows:
        break
    row = plot_idx // n_cols
    col = plot_idx % n_cols

    img_path = test_image_paths[idx]
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0

    axes[row, col].imshow(np.clip(img_array, 0, 1))
    true_label = class_names[y_true[idx]]
    pred_label = class_names[y_pred[idx]]
    axes[row, col].set_title(f'✓ {pred_label}', color='green', fontweight='bold', fontsize=10)
    axes[row, col].axis('off')
    plot_idx += 1

# Add some incorrect predictions
for idx in incorrect_indices[:n_cols * n_rows // 2]:
    if plot_idx >= n_cols * n_rows:
        break
    row = plot_idx // n_cols
    col = plot_idx % n_cols

    img_path = test_image_paths[idx]
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0

    axes[row, col].imshow(np.clip(img_array, 0, 1))
    true_label = class_names[y_true[idx]]
    pred_label = class_names[y_pred[idx]]
    axes[row, col].set_title(f'✗ {pred_label}\n(True: {true_label})', color='red', fontweight='bold', fontsize=9)
    axes[row, col].axis('off')
    plot_idx += 1

plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Sample predictions plot saved as 'sample_predictions.png'")

# ==================== VISUALIZATION 6: METRICS SUMMARY DASHBOARD ====================
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

fig.suptitle('Summary Metrics Dashboard', fontsize=16, fontweight='bold')

# Overall accuracy
ax1 = fig.add_subplot(gs[0, 0])
ax1.text(0.5, 0.7, f'{test_acc:.2%}', ha='center', va='center', fontsize=48, fontweight='bold', color='#1f77b4')
ax1.text(0.5, 0.2, 'Overall Accuracy', ha='center', va='center', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# Macro vs Weighted
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

ax2 = fig.add_subplot(gs[0, 1])
metrics = ['Precision', 'Recall', 'F1-Score']
macro_vals = [precision_macro, recall_macro, f1_macro]
weighted_vals = [precision_w, recall_w, f1_w]
x = np.arange(len(metrics))
width = 0.35
ax2.bar(x - width/2, macro_vals, width, label='Macro', alpha=0.8)
ax2.bar(x + width/2, weighted_vals, width, label='Weighted', alpha=0.8)
ax2.set_ylabel('Score')
ax2.set_title('Macro vs Weighted Metrics', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend()
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3, axis='y')

# Per-class recall (sensitivity)
ax3 = fig.add_subplot(gs[1, :])
precision_pc, recall_pc, f1_pc, support = precision_recall_fscore_support(y_true, y_pred, average=None)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
ax3.barh(class_names, recall_pc, color=colors[:len(class_names)], alpha=0.8)
ax3.set_xlabel('Recall (Sensitivity)')
ax3.set_title('Per-Class Recall (Sensitivity)', fontweight='bold')
ax3.set_xlim([0, 1])
for i, v in enumerate(recall_pc):
    ax3.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# Class distribution with predictions
ax4 = fig.add_subplot(gs[2, 0])
class_counts = np.bincount(y_true)
ax4.bar(class_names, class_counts, alpha=0.7, color='lightblue', label='Total Samples')
ax4.set_ylabel('Count')
ax4.set_title('Class Distribution', fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
for i, v in enumerate(class_counts):
    ax4.text(i, v + 2, str(v), ha='center', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Summary statistics
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
n_test_samples = len(y_true)
n_correct = np.sum(correct_mask)
n_incorrect = np.sum(~correct_mask)

summary_text = f"""
SUMMARY STATISTICS

Total Test Samples: {n_test_samples}
Correct Predictions: {n_correct}
Incorrect Predictions: {n_incorrect}

Test Accuracy: {test_acc:.4f}
Test Loss: {test_loss:.4f}

Macro Precision: {precision_macro:.4f}
Macro Recall: {recall_macro:.4f}
Macro F1-Score: {f1_macro:.4f}

Weighted Precision: {precision_w:.4f}
Weighted Recall: {recall_w:.4f}
Weighted F1-Score: {f1_w:.4f}
"""

ax5.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('metrics_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Metrics dashboard saved as 'metrics_dashboard.png'")

print("\n" + "="*60)
print("ALL VISUALIZATIONS SAVED SUCCESSFULLY!")
print("="*60)
print("\nGenerated files:")
print("  1. class_distribution.png")
print("  2. augmented_samples.png")
print("  3. confusion_matrix.png")
print("  4. training_history.png")
print("  5. sample_predictions.png")
print("  6. metrics_dashboard.png")
print("="*60 + "\n")

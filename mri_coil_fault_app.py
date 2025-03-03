
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import logging
from pathlib import Path
import io
import pydicom
from pydicom.dataset import Dataset, FileDataset
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tensorflow as tf
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import io
from PIL import Image


# Set up logging
logging.basicConfig(filename='mri_app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define intensity ranges with EXTREME differences between classes
# These need to be drastically different for clear classification
NORMAL_INTENSITY = 150
CRACK_INTENSITY = 2500  # Much higher to be very distinct
WEAR_INTENSITY = 2000   # Also much higher
BACKGROUND_INTENSITY = 100

def generate_synthetic_dcm(num_samples=50, img_size=32, base_path="mri_coil_data"):
    """
    Generate synthetic DICOM images with EXTREME differences between classes.
    This ensures the model can clearly distinguish between fault types.
    """
    os.makedirs(base_path, exist_ok=True)
    images = []
    labels = []
    filenames = []
    
    # Create synthetic images for each class with extreme differences
    for i in range(num_samples):
        # 1. NORMAL - very uniform, low intensity
        normal_img = np.ones((img_size, img_size), dtype=np.float32) * NORMAL_INTENSITY
        # Add minimal noise
        normal_img += np.random.normal(0, 5, (img_size, img_size))
        normal_img = np.clip(normal_img, NORMAL_INTENSITY-10, NORMAL_INTENSITY+10)
        images.append(normal_img)
        labels.append(0)
        filenames.append(f"normal_{i}.dcm")
        
        # 2. CRACK - extremely prominent diagonal line
        crack_img = np.ones((img_size, img_size), dtype=np.float32) * BACKGROUND_INTENSITY
        # Add a very clear diagonal crack
        crack_width = 3
        for j in range(img_size):
            if j < img_size - crack_width:
                for w in range(crack_width):
                    crack_img[j, j+w] = CRACK_INTENSITY
        images.append(crack_img)
        labels.append(1)
        filenames.append(f"crack_{i}.dcm")
        
        # 3. WEAR - many very bright spots
        wear_img = np.ones((img_size, img_size), dtype=np.float32) * BACKGROUND_INTENSITY
        # Add many high-intensity wear spots
        num_spots = 25  # More spots
        for _ in range(num_spots):
            x = np.random.randint(0, img_size)
            y = np.random.randint(0, img_size)
            spot_size = np.random.randint(3, 5)
            for dx in range(-spot_size, spot_size+1):
                for dy in range(-spot_size, spot_size+1):
                    if 0 <= x+dx < img_size and 0 <= y+dy < img_size:
                        distance = np.sqrt(dx*dx + dy*dy)
                        if distance <= spot_size:
                            wear_img[x+dx, y+dy] = WEAR_INTENSITY
        images.append(wear_img)
        labels.append(2)
        filenames.append(f"wear_{i}.dcm")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Shuffle data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    filenames = np.array(filenames)[indices].tolist()
    
    # Save raw pixel data before normalization (for DICOM files)
    X_raw = X.copy()
    
    # Normalize data for model training (0-1 range)
    X_min, X_max = np.min(X), np.max(X)
    X = (X - X_min) / (X_max - X_min)
    X = X[..., np.newaxis]  # Add channel dimension
    
    # Save normalized data for model training
    np.savez(f"{base_path}/data.npz", X=X, y=y, filenames=filenames, X_min=X_min, X_max=X_max)
    
    # Save metadata for future loading
    with open(f"{base_path}/metadata.txt", "w") as f:
        f.write(f"X_min: {X_min}\n")
        f.write(f"X_max: {X_max}\n")
        f.write(f"NORMAL_INTENSITY: {NORMAL_INTENSITY}\n")
        f.write(f"CRACK_INTENSITY: {CRACK_INTENSITY}\n")
        f.write(f"WEAR_INTENSITY: {WEAR_INTENSITY}\n")
        f.write(f"BACKGROUND_INTENSITY: {BACKGROUND_INTENSITY}\n")
    
    # Create DICOM files
    for i, (img, fname) in enumerate(zip(X_raw, filenames)):
        ds = Dataset()
        ds.Rows = img_size
        ds.Columns = img_size
        
        # Convert to 16-bit unsigned integers for DICOM
        pixel_data = img.astype(np.uint16)
        ds.PixelData = pixel_data.tobytes()
        
        # Set DICOM attributes
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Modality = "MR"
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        
        # Create metadata
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Save DICOM file
        dicom_file = FileDataset(f"{base_path}/{fname}", ds, file_meta=file_meta, preamble=b"\0" * 128)
        dicom_file.save_as(f"{base_path}/{fname}")
    
    return X, y, filenames, X_min, X_max

def load_metadata(base_path="mri_coil_data"):
    """Load normalization parameters from metadata file"""
    metadata = {}
    try:
        with open(f"{base_path}/metadata.txt", "r") as f:
            for line in f:
                key, value = line.strip().split(": ")
                metadata[key] = float(value)
        return metadata
    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        # Return default values if file doesn't exist
        return {
            "X_min": 100,
            "X_max": 2500,
            "NORMAL_INTENSITY": 150,
            "CRACK_INTENSITY": 2500,
            "WEAR_INTENSITY": 2000,
            "BACKGROUND_INTENSITY": 100
        }

def build_balanced_cnn_model(input_shape=(32, 32, 1), num_classes=3):
    """
    Build a CNN model specifically designed for this task with:
    - Class weights to handle imbalance
    - Reduced complexity to prevent overfitting
    - Regularization to improve generalization
    """
    model = models.Sequential([
        # First convolutional block with regularization
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape, name='conv1',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Fully connected layers with stronger regularization
        layers.Flatten(name='flatten'),
        layers.Dense(64, activation='relu', name='dense1',
                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(name='bn3'),
        layers.Dropout(0.4, name='dropout'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    # Compile with a lower learning rate for stability
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_balanced_model(X_train, y_train, X_test, y_test, epochs=15, batch_size=16):
    """
    Train the model with techniques to address class imbalance:
    - Compute class weights
    - Use early stopping
    - Apply learning rate schedule
    """
    # Calculate class weights to handle imbalance
    # This gives more importance to underrepresented classes
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weights = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
    
    logging.info(f"Class weights: {class_weights}")
    
    # Build model
    model = build_balanced_cnn_model()
    
    # Set up callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # Train with class weights
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Test accuracy: {test_acc:.4f}")
    
    return model, history

def get_image_features(image):
    """Extract key features from image for fault detection"""
    if len(image.shape) == 3 and image.shape[2] == 1:
        img = image[:, :, 0]  # Remove channel dimension
    else:
        img = image
        
    features = {}
    
    # Basic statistics
    features["mean"] = np.mean(img)
    features["std"] = np.std(img)
    features["min"] = np.min(img)
    features["max"] = np.max(img)
    
    # Diagonal features (for crack detection)
    diagonal = np.array([img[i, i] for i in range(min(img.shape))])
    features["diagonal_mean"] = np.mean(diagonal)
    features["diagonal_max"] = np.max(diagonal)
    
    # Off-diagonal maximum (for comparison with diagonal)
    mask = ~np.eye(img.shape[0], dtype=bool)
    features["off_diagonal_max"] = np.max(img * mask)
    
    # High intensity regions (for wear detection)
    features["high_intensity_ratio"] = np.sum(img > 0.6) / img.size
    
    # Intensity distribution
    features["percentile_25"] = np.percentile(img, 25)
    features["percentile_75"] = np.percentile(img, 75)
    
    # Entropy - measure of disorder (higher for wear patterns)
    hist, _ = np.histogram(img, bins=20, range=(0, 1))
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    features["entropy"] = entropy
    
    return features

def predict_fault_direct(model, image):
    """
    Use direct model prediction without rule overrides.
    The model should be properly trained to handle this directly.
    """
    try:
        # Get model prediction
        pred = model.predict(image[np.newaxis, ...], verbose=0)[0]
        fault_types = ["Normal", "Crack Detected", "Wear Detected"]
        pred_idx = np.argmax(pred)
        confidence = pred[pred_idx]
        
        # Log prediction
        logging.info(f"Prediction: {fault_types[pred_idx]} with confidence {confidence:.4f}")
        logging.info(f"Probabilities: Normal={pred[0]:.4f}, Crack={pred[1]:.4f}, Wear={pred[2]:.4f}")
        
        # Extract features for information only
        features = get_image_features(image)
        logging.info(f"Key features: diagonal_max={features['diagonal_max']:.4f}, "
                    f"high_intensity_ratio={features['high_intensity_ratio']:.4f}, "
                    f"std={features['std']:.4f}, entropy={features['entropy']:.4f}")
        
        return fault_types[pred_idx], confidence, pred
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        logging.error(f"Prediction error: {str(e)}")
        return "Error", 0.0, np.zeros(3)

def verify_manually(image):
    """
    A separate, simple rule-based verification as a backup.
    Independent of the model.
    """
    features = get_image_features(image)
    
    # Very simple, clear-cut rules for visual patterns
    if features["diagonal_max"] > 0.8 and features["diagonal_max"] > 1.5 * features["off_diagonal_max"]:
        return "Crack Fault Confirmed"
    elif features["high_intensity_ratio"] > 0.1 and features["entropy"] > 3.0:
        return "Wear Fault Confirmed"
    elif features["std"] < 0.05 and features["max"] - features["min"] < 0.2:
        return "No Fault Detected - Normal"
    else:
        return "Indeterminate - Inspect Manually"

def evaluate_model_with_confusion_matrix(model, X_test, y_test):
    """
    Evaluate model with detailed metrics including a confusion matrix.
    Helps identify which classes are being confused.
    """
    # Get predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Basic metrics
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Confusion matrix
    cm = tf.math.confusion_matrix(y_test, y_pred_classes).numpy()
    
    # Per-class metrics
    classes = ["Normal", "Crack", "Wear"]
    class_metrics = {}
    for i, class_name in enumerate(classes):
        true_positive = cm[i, i]
        false_negative = np.sum(cm[i, :]) - true_positive
        false_positive = np.sum(cm[:, i]) - true_positive
        
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * (precision * recall) / max(1e-10, precision + recall)
        
        class_metrics[class_name] = {
            "precision": precision, 
            "recall": recall,
            "f1": f1,
            "count": np.sum(y_test == i)
        }
    
    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": cm,
        "class_metrics": class_metrics
    }

def visualize_layers_robust(model, image, selected_layers):
    """
    Extremely robust visualization function that works with any Keras model structure
    and handles all edge cases.
    """
    try:
        # First forcibly build the model if it's not built
        if not model.built:
            logging.info("Model not built, building it now")
            model.build(image[np.newaxis, ...].shape)
            
        # Force a prediction to ensure all layers are properly initialized
        try:
            _ = model.predict(image[np.newaxis, ...], verbose=0)
        except:
            # If prediction fails, try forward pass instead
            logging.info("Model prediction failed, trying call method")
            _ = model(image[np.newaxis, ...])
            
        # Get actual layer names from the model
        actual_layer_names = [layer.name for layer in model.layers]
        logging.info(f"Available layers: {actual_layer_names}")
        
        # Find valid layers from selected options
        valid_layers = [name for name in selected_layers if name in actual_layer_names]
        logging.info(f"Valid selected layers: {valid_layers}")
        
        if not valid_layers:
            st.warning(f"None of the selected layers ({selected_layers}) exist in this model. Available layers are: {actual_layer_names}")
            return []
        
        figures = []
        for layer_name in valid_layers:
            try:
                # Get the actual layer object
                layer = model.get_layer(layer_name)
                logging.info(f"Processing layer: {layer_name}, type: {type(layer).__name__}")
                
                # For layers that don't have meaningful activations, just show info
                if isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.BatchNormalization)):
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.text(0.5, 0.5, f"Layer '{layer_name}' ({type(layer).__name__}) does not have meaningful visualizations", 
                           horizontalalignment='center', verticalalignment='center')
                    ax.axis('off')
                    figures.append(fig)
                    continue
                
                # Alternative approach: use a custom function to extract outputs
                # This is the most robust way that works with any model structure
                
                # Create a simple function to extract outputs
                @tf.function
                def get_layer_output(input_data):
                    # Create a model that goes from model input to the layer we want
                    feature_extractor = tf.keras.Model(
                        inputs=model.inputs,
                        outputs=layer.output,
                        name=f"feature_extractor_{layer_name}"
                    )
                    # Get output for this layer
                    return feature_extractor(input_data)
                
                # Get output for this layer
                layer_output = get_layer_output(image[np.newaxis, ...]).numpy()
                
                # Create visualization based on output shape
                if len(layer_output.shape) == 4:  # Conv layer [batch, height, width, channels]
                    n_features = min(16, layer_output.shape[3])
                    size = int(np.ceil(np.sqrt(n_features)))
                    
                    fig, axs = plt.subplots(size, size, figsize=(10, 10))
                    fig.suptitle(f"Layer: {layer_name} - Feature Maps")
                    
                    # Handle case where size is 1 (single image)
                    if size == 1:
                        # Handle single feature case
                        if n_features == 1:
                            axs.imshow(layer_output[0, :, :, 0], cmap='viridis')
                            axs.set_xticks([])
                            axs.set_yticks([])
                        else:
                            # Multiple features but only one row/column in grid
                            axs = np.array([axs])
                            for i in range(n_features):
                                if i < len(axs):
                                    axs[i].imshow(layer_output[0, :, :, i], cmap='viridis')
                                    axs[i].set_xticks([])
                                    axs[i].set_yticks([])
                    else:
                        # Multiple features in a grid
                        axs = axs.flatten()
                        for i in range(n_features):
                            if i < len(axs):
                                axs[i].imshow(layer_output[0, :, :, i], cmap='viridis')
                                axs[i].set_xticks([])
                                axs[i].set_yticks([])
                        
                        # Hide unused subplots
                        for i in range(n_features, len(axs)):
                            axs[i].axis('off')
                    
                    plt.tight_layout()
                    figures.append(fig)
                    
                elif len(layer_output.shape) == 2:  # Dense layer [batch, units]
                    fig, ax = plt.subplots(figsize=(10, 4))
                    im = ax.imshow(layer_output[0].reshape(1, -1), cmap='viridis', aspect='auto')
                    ax.set_title(f"Layer: {layer_name} - Neuron Activations")
                    plt.colorbar(im, ax=ax)
                    figures.append(fig)
                    
                else:  # Other layers with different shapes
                    # Try to reshape to 2D for visualization
                    flattened = layer_output.reshape(1, -1)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    im = ax.imshow(flattened, cmap='viridis', aspect='auto')
                    ax.set_title(f"Layer: {layer_name} - Reshaped Output")
                    plt.colorbar(im, ax=ax)
                    figures.append(fig)
                    
            except Exception as e:
                # If a specific layer fails, log it but continue with others
                logging.error(f"Error visualizing layer {layer_name}: {str(e)}")
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.text(0.5, 0.5, f"Could not visualize layer '{layer_name}': {str(e)}", 
                       horizontalalignment='center', verticalalignment='center')
                ax.axis('off')
                figures.append(fig)
                continue
                
        return figures
        
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        logging.error(f"Visualization error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return []

def display_confusion_matrix(cm, classes=["Normal", "Crack", "Wear"]):
    """
    Create and display a confusion matrix visualization.
    Helps identify which classes are being confused.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=classes, yticklabels=classes, ax=ax)
    
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    
    return fig

def load_dcm(file, metadata=None):
    """Load and normalize a DICOM file for prediction"""
    if metadata is None:
        metadata = load_metadata()
        
    try:
        # Read DICOM file
        ds = pydicom.dcmread(file)
        img = ds.pixel_array.astype(np.float32)
        
        # Log raw pixel statistics
        logging.info(f"Raw DICOM stats - Min: {np.min(img)}, Max: {np.max(img)}, Mean: {np.mean(img)}")
        
        # Normalize using the same parameters as training data
        img = (img - metadata["X_min"]) / (metadata["X_max"] - metadata["X_min"])
        img = np.clip(img, 0.0, 1.0)
        
        # Resize to model input size if needed
        current_size = img.shape
        if current_size[0] != 32 or current_size[1] != 32:
            img = tf.image.resize(img[..., np.newaxis], [32, 32], method='bilinear').numpy()
        else:
            img = img[..., np.newaxis]  # Add channel dimension
            
        # Log normalized statistics
        logging.info(f"Normalized DICOM stats - Min: {np.min(img)}, Max: {np.max(img)}, Mean: {np.mean(img)}")
            
        return img
    except Exception as e:
        st.error(f"Failed to load DICOM: {e}")
        logging.error(f"DICOM load error: {str(e)}")
        return None
    

class NeuralActivityCallback(tf.keras.callbacks.Callback):
    """Callback to update neural activity visualization during training"""
    def __init__(self, visualizer, placeholder, layer_name=None, update_freq=5):
        super().__init__()
        self.visualizer = visualizer
        self.placeholder = placeholder
        self.layer_name = layer_name
        self.update_freq = update_freq
        self.batch_count = 0
        
    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % self.update_freq == 0:
            # Get a sample from the validation data to visualize
            if hasattr(self.model, 'validation_data'):
                x_sample = self.model.validation_data[0][:1]  # Take first sample
            else:
                # Fallback to training data
                x_sample = self.model._x[:1]  # Take first sample
                
            # Update visualizer with new activations
            self.visualizer.get_layer_activations(x_sample)
            
            # Create visualization
            fig = self.visualizer.create_firing_visualization(self.layer_name)
            if fig:
                # Convert figure to image
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                
                # Update placeholder
                self.placeholder.image(img, caption=f"Batch {self.batch_count}", use_column_width=True)
                plt.close(fig)


class NeuralActivityVisualizer:
    """
    Class to visualize neural network activity during training
    like neurons firing in a brain.
    """
    def __init__(self, model):
        self.model = model
        self.layer_outputs = {}
        self.prev_outputs = {}
        self.firing_state = {}
        self.activity_history = {}
        self.max_history = 10  # Number of frames to keep in history
        
        # Custom colormap for neuron firing (dark blue to bright yellow)
        self.cmap = LinearSegmentedColormap.from_list('neural_firing', 
                                                  ['#000033', '#0000FF', '#00FFFF', '#FFFF00', '#FFFFFF'], 
                                                  N=256)
    
    def setup_visualizer(self):
        """Set up visualization for layers in the model"""
        # Create a modified model that outputs activations from all layers
        layer_outputs = []
        layer_names = []
        
        for layer in self.model.layers:
            # Skip layers that don't have activations we can visualize
            if isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.InputLayer)):
                continue
                
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)
        
        # Create a model that returns all layer outputs
        self.activation_model = tf.keras.Model(inputs=self.model.inputs, 
                                         outputs=layer_outputs)
        self.layer_names = layer_names
        
        # Initialize previous outputs and firing state for each layer
        for name in self.layer_names:
            self.prev_outputs[name] = None
            self.firing_state[name] = None
            self.activity_history[name] = []
    
    def get_layer_activations(self, input_data):
        """Get activations for all layers given input data"""
        activations = self.activation_model.predict(input_data, verbose=0)
        
        # Handle case when only one layer is output
        if not isinstance(activations, list):
            activations = [activations]
            
        # Update layer outputs
        for i, name in enumerate(self.layer_names):
            self.layer_outputs[name] = activations[i]
            
            # Calculate firing state (change in activation)
            if self.prev_outputs[name] is not None:
                # Calculate absolute differences to show "firing"
                diff = np.abs(activations[i] - self.prev_outputs[name])
                # Apply threshold to highlight significant changes
                self.firing_state[name] = np.where(diff > 0.1, diff, 0)
            else:
                # First run - use absolute values
                self.firing_state[name] = np.abs(activations[i])
                
            # Update history
            self.activity_history[name].append(self.firing_state[name].copy())
            if len(self.activity_history[name]) > self.max_history:
                self.activity_history[name].pop(0)
                
            # Update previous outputs for next iteration
            self.prev_outputs[name] = activations[i].copy()
    
    def create_firing_visualization(self, layer_name=None):
        """Create visualization of neurons firing for a specific layer"""
        if layer_name is None:
            # Default to first visualization-friendly layer
            for name in self.layer_names:
                if len(self.activity_history[name]) > 0:
                    layer_name = name
                    break
            if layer_name is None:
                return None
        
        if len(self.activity_history[layer_name]) == 0:
            return None
            
        # Get the firing state for this layer
        history = self.activity_history[layer_name]
        latest = history[-1]
        
        # Prepare visualization based on layer type
        if len(latest.shape) == 4:  # Conv layer [batch, height, width, channels]
            return self._visualize_conv_layer(layer_name, history)
        elif len(latest.shape) == 2:  # Dense layer [batch, units]
            return self._visualize_dense_layer(layer_name, history)
        else:
            # Try to reshape to 2D
            return self._visualize_generic_layer(layer_name, history)
    
    def _visualize_conv_layer(self, layer_name, history):
        """Visualize a convolutional layer as a grid of firing neurons"""
        latest = history[-1]
        batch, height, width, channels = latest.shape
        
        # Create a combined visualization of all channels
        combined = np.zeros((height, width))
        decay_factor = 1.0
        
        # Combine multiple frames with decay to create trail effect
        for frame in reversed(history):
            # Maximum across channels and batch dimension
            frame_max = np.max(np.max(frame, axis=0), axis=2)
            combined += frame_max * decay_factor
            decay_factor *= 0.7  # Decay for older frames
        
        # Normalize for visualization
        combined = np.clip(combined, 0, 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(combined, cmap=self.cmap, interpolation='nearest')
        ax.set_title(f"Neural Activity - {layer_name}")
        plt.colorbar(im, ax=ax, label="Activation Strength")
        
        return fig
    
    def _visualize_dense_layer(self, layer_name, history):
        """Visualize a dense layer as a grid of firing neurons"""
        latest = history[-1]
        batch, units = latest.shape
        
        # Create a combined visualization with decay
        combined = np.zeros(units)
        decay_factor = 1.0
        
        for frame in reversed(history):
            # Take first batch example
            frame_data = frame[0]
            combined += frame_data * decay_factor
            decay_factor *= 0.7
            
        # Normalize and reshape to a grid for better visualization
        combined = np.clip(combined, 0, 1)
        grid_size = int(np.ceil(np.sqrt(units)))
        grid = np.zeros((grid_size, grid_size))
        for i in range(min(units, grid_size * grid_size)):
            row, col = i // grid_size, i % grid_size
            grid[row, col] = combined[i]
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(grid, cmap=self.cmap, interpolation='nearest')
        ax.set_title(f"Neural Activity - {layer_name}")
        plt.colorbar(im, ax=ax, label="Activation Strength")
        
        return fig
    
    def _visualize_generic_layer(self, layer_name, history):
        """Visualize any other layer type by reshaping to 2D"""
        latest = history[-1]
        
        # Flatten and reshape to a 2D grid
        flattened = latest.reshape(latest.shape[0], -1)
        batch, features = flattened.shape
        
        # Create a combined visualization with decay
        combined = np.zeros(features)
        decay_factor = 1.0
        
        for frame in reversed(history):
            frame_flat = frame.reshape(frame.shape[0], -1)
            combined += frame_flat[0] * decay_factor
            decay_factor *= 0.7
            
        # Reshape to a grid
        grid_size = int(np.ceil(np.sqrt(features)))
        grid = np.zeros((grid_size, grid_size))
        for i in range(min(features, grid_size * grid_size)):
            row, col = i // grid_size, i % grid_size
            grid[row, col] = combined[i]
            
        # Normalize for visualization
        grid = np.clip(grid / np.max(grid) if np.max(grid) > 0 else grid, 0, 1)
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(grid, cmap=self.cmap, interpolation='nearest')
        ax.set_title(f"Neural Activity - {layer_name}")
        plt.colorbar(im, ax=ax, label="Activation Strength")
        
        return fig
    

def train_balanced_model_with_visualization(X_train, y_train, X_test, y_test, epochs=15, batch_size=16):
    """
    Train the model with techniques to address class imbalance AND brain visualization.
    This preserves all the key aspects of the original train_balanced_model function.
    """
    st.subheader("Neural Network Training with Brain Visualization")
    st.write("Watch the neural activity in real-time as the network trains - just like neurons firing in a brain!")
    
    # Calculate class weights to handle imbalance
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weights = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
    
    st.write("Using class weights:", class_weights)
    logging.info(f"Class weights: {class_weights}")
    
    # Build model
    model = build_balanced_cnn_model()
    
    # Create placeholders for visualizations
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    activity_placeholder = st.empty() 
    chart_placeholder = st.empty()
    
    # Set up the neural activity visualizer
    visualizer = NeuralActivityVisualizer(model)
    visualizer.setup_visualizer()
    
    # Show model summary
    with st.expander("Model Architecture"):
        summary_str = []
        model.summary(print_fn=lambda x: summary_str.append(x))
        st.code('\n'.join(summary_str))
    
    # Create a progress bar
    progress_bar = progress_placeholder.progress(0.0)
    
    # Create a brain visualization callback
    class BrainVisualizationCallback(tf.keras.callbacks.Callback):
        def __init__(self, visualizer, activity_placeholder, metrics_placeholder, progress_bar, chart_placeholder):
            self.visualizer = visualizer
            self.activity_placeholder = activity_placeholder
            self.metrics_placeholder = metrics_placeholder
            self.progress_bar = progress_bar
            self.chart_placeholder = chart_placeholder
            self.sample_data = X_train[:1]  # Use first training example for visualization
            
        def on_epoch_begin(self, epoch, logs=None):
            pass
            
        def on_epoch_end(self, epoch, logs=None):
            # Update progress
            progress = (epoch + 1) / epochs
            self.progress_bar.progress(progress)
            
            # Update metrics
            metrics_text = f"Epoch {epoch+1}/{epochs}"
            for k, v in logs.items():
                metrics_text += f" - {k}: {v:.4f}"
            self.metrics_placeholder.write(metrics_text)
            
            # Update visualization
            self.visualizer.get_layer_activations(self.sample_data)
            fig = self.visualizer.create_firing_visualization()
            if fig:
                self.activity_placeholder.pyplot(fig)
                plt.close(fig)
                
            # Update charts if we have training history
            if hasattr(self.model, 'history') and self.model.history is not None:
                history = self.model.history.history
                if len(history.get('loss', [])) > 1:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Loss curve
                    ax1.plot(history['loss'], label='Training Loss')
                    if 'val_loss' in history:
                        ax1.plot(history['val_loss'], label='Validation Loss')
                    ax1.set_title("Loss Over Epochs")
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Loss")
                    ax1.legend()
                    
                    # Accuracy curve
                    ax2.plot(history['accuracy'], label='Training Accuracy')
                    if 'val_accuracy' in history:
                        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
                    ax2.set_title("Accuracy Over Epochs")
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Accuracy")
                    ax2.legend()
                    
                    self.chart_placeholder.pyplot(fig)
                    plt.close(fig)
            
        def on_batch_end(self, batch, logs=None):
            # Update brain visualization occasionally during training
            if batch % 10 == 0:
                self.visualizer.get_layer_activations(self.sample_data)
                fig = self.visualizer.create_firing_visualization()
                if fig:
                    self.activity_placeholder.pyplot(fig)
                    plt.close(fig)
    
    # Initialize visualizer with a sample
    visualizer.get_layer_activations(X_train[:1])
    fig = visualizer.create_firing_visualization()
    if fig:
        activity_placeholder.pyplot(fig)
        plt.close(fig)
    
    # Set up callbacks - KEEP THE ORIGINAL CALLBACKS that were working well
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        ),
        # Add our brain visualization callback
        BrainVisualizationCallback(
            visualizer=visualizer,
            activity_placeholder=activity_placeholder,
            metrics_placeholder=metrics_placeholder,
            progress_bar=progress_bar,
            chart_placeholder=chart_placeholder
        )
    ]
    
    # Train with class weights - USING THE SAME APPROACH as the original function
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,  # Use the same validation split approach
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=0  # Set to 0 to avoid cluttering the display
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Test accuracy: {test_acc:.4f}")
    progress_placeholder.success(f"Training Complete! Test Accuracy: {test_acc:.4f}")
    
    return model, history
def create_animated_brain_visualization(model, X_train, y_train, epochs=10, batch_size=32):
    """
    Create an animated brain-like visualization of the neural network during training.
    
    Parameters:
    - model: The neural network model
    - X_train, y_train: Training data
    - epochs: Number of training epochs
    - batch_size: Training batch size
    
    Returns:
    - Trained model
    """
    # Set up the neural activity visualizer
    visualizer = NeuralActivityVisualizer(model)
    visualizer.setup_visualizer()
    
    # Create placeholders for the visualizations
    activity_placeholder = st.empty()
    progress_text = st.empty()
    progress_bar = st.progress(0.0)
    
    # Get a sample batch for initial visualization
    x_sample = X_train[:1]
    visualizer.get_layer_activations(x_sample)
    
    # Show initial activity visualization
    fig = visualizer.create_firing_visualization()
    if fig:
        activity_placeholder.pyplot(fig)
        plt.close(fig)
    
    # Create callback for updating visualization
    callback = NeuralActivityCallback(
        visualizer=visualizer, 
        placeholder=activity_placeholder,
        update_freq=max(1, int(len(X_train) / batch_size / 10))  # Update ~10 times per epoch
    )
    
    # Train the model with the visualization callback
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[callback],
        verbose=0
    )
    
    # Clean up
    progress_bar.progress(1.0)
    progress_text.text("Training complete!")
    
    return model, history

def train_with_brain_visualization(X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    """
    Train a model with real-time brain-like visualization of neural activations.
    
    Parameters:
    - X_train, y_train: Training data
    - X_test, y_test: Testing data
    - epochs: Number of training epochs
    - batch_size: Batch size for training
    
    Returns:
    - trained model and history
    """
    st.subheader("Neural Network Training with Brain-like Visualization")
    st.write("Watch the neural activity in real-time as the network trains - just like neurons firing in a brain!")
    
    # Display training parameters
    st.write(f"Training with {len(X_train)} samples, {epochs} epochs, batch size {batch_size}")
    
    # Calculate class weights to handle imbalance
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weights = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
    st.write("Class weights:", class_weights)
    
    # Build model
    model = build_balanced_cnn_model()
    
    # Create placeholders for visualizations
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    activity_placeholder = st.empty()
    loss_placeholder = st.empty()
    
    # Set up the neural activity visualizer
    visualizer = NeuralActivityVisualizer(model)
    visualizer.setup_visualizer()
    
    # Show model summary
    with st.expander("Model Architecture"):
        # Capture model summary
        summary_str = []
        model.summary(print_fn=lambda x: summary_str.append(x))
        st.code('\n'.join(summary_str))
    
    # Initialize progress bar
    progress_bar = progress_placeholder.progress(0.0)
    
    # Initialize history tracking
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    total_batches = int(np.ceil(len(X_train) / batch_size)) * epochs
    batch_count = 0
    
    # Prepare validation data
    val_split = 0.2
    val_idx = int(len(X_train) * (1 - val_split))
    x_val, y_val = X_train[val_idx:], y_train[val_idx:]
    X_train, y_train = X_train[:val_idx], y_train[:val_idx]
    
    # Update frequency for the visualization
    update_freq = max(1, int(total_batches / 30))  # Update ~30 times during training
    
    # Initialize visualization with a sample
    x_sample = X_train[:1]
    visualizer.get_layer_activations(x_sample)
    fig = visualizer.create_firing_visualization()
    if fig:
        activity_placeholder.pyplot(fig)
        plt.close(fig)
    
    # Training loop with visualization updates
    for epoch in range(epochs):
        # Shuffle training data for each epoch
        indices = np.random.permutation(len(X_train))
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]
        
        # Process each batch
        for i in range(0, len(X_train_shuffled), batch_size):
            batch_count += 1
            
            # Get batch data
            end = min(i + batch_size, len(X_train_shuffled))
            x_batch = X_train_shuffled[i:end]
            y_batch = y_train_shuffled[i:end]
            
            # Train on batch
            batch_logs = model.train_on_batch(
                x_batch, y_batch, 
                class_weight=class_weights,
                return_dict=True
            )
            
            # Update history
            if 'loss' in batch_logs:
                history['loss'].append(batch_logs['loss'])
            if 'accuracy' in batch_logs:
                history['accuracy'].append(batch_logs['accuracy'])
            
            # Periodically validate
            if batch_count % (total_batches // (epochs * 2)) == 0 or batch_count == total_batches:
                val_logs = model.evaluate(x_val, y_val, verbose=0)
                if isinstance(val_logs, list):
                    val_loss, val_acc = val_logs
                else:
                    val_loss, val_acc = val_logs, None
                
                history['val_loss'].append(val_loss)
                if val_acc is not None:
                    history['val_accuracy'].append(val_acc)
                
                # Update metrics display
                metrics_text = f"Epoch {epoch+1}/{epochs} - Batch {batch_count}/{total_batches} - "
                metrics_text += f"Loss: {batch_logs.get('loss', 0):.4f}"
                if 'accuracy' in batch_logs:
                    metrics_text += f" - Accuracy: {batch_logs['accuracy']:.4f}"
                metrics_text += f" - Val Loss: {val_loss:.4f}"
                if val_acc is not None:
                    metrics_text += f" - Val Accuracy: {val_acc:.4f}"
                
                metrics_placeholder.write(metrics_text)
                
                # Update loss chart
                if len(history['loss']) > 1 and len(history['val_loss']) > 1:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(history['loss'], label='Training Loss')
                    # Use a separate x-axis for validation (since it's evaluated less frequently)
                    val_x = np.linspace(0, len(history['loss'])-1, len(history['val_loss']))
                    ax.plot(val_x, history['val_loss'], label='Validation Loss')
                    ax.set_title("Loss Over Training")
                    ax.set_xlabel("Batch")
                    ax.set_ylabel("Loss")
                    ax.legend()
                    loss_placeholder.pyplot(fig)
                    plt.close(fig)
            
            # Update neural activity visualization
            if batch_count % update_freq == 0 or batch_count == total_batches:
                # Get activations for the first validation sample
                visualizer.get_layer_activations(x_val[:1])
                
                # Create and display visualization
                fig = visualizer.create_firing_visualization()
                if fig:
                    activity_placeholder.pyplot(fig)
                    plt.close(fig)
            
            # Update progress
            progress = batch_count / total_batches
            progress_bar.progress(progress)
    
    # Final evaluation
    test_logs = model.evaluate(X_test, y_test, verbose=0)
    if isinstance(test_logs, list):
        test_loss, test_acc = test_logs
    else:
        test_loss, test_acc = test_logs, None
    
    progress_placeholder.success(f"Training Complete! Test Loss: {test_loss:.4f}" + 
                               (f" - Test Accuracy: {test_acc:.4f}" if test_acc is not None else ""))
    
    # Convert history to expected format for compatibility
    history_obj = type('obj', (object,), {'history': history})
    
    return model, history_obj


def main():
    """Main Streamlit app with modified training and evaluation approach"""
    st.set_page_config(page_title="MRI Coil Fault Detector", layout="wide")
    st.title("MRI Coil Fault Detector")
    st.markdown("**Advanced AI Diagnostic Tool** - Detect MRI coil faults with precision")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Data options
    st.sidebar.subheader("Data Options")
    regenerate_data = st.sidebar.checkbox("Regenerate synthetic data")
    
    if regenerate_data:
        if st.sidebar.button("Generate New Data"):
            with st.spinner("Generating new synthetic data with extreme class differences..."):
                try:
                    # Remove existing data
                    if os.path.exists("mri_coil_data/data.npz"):
                        os.remove("mri_coil_data/data.npz")
                    
                    # Generate new data
                    X_data, y_data, filenames, X_min, X_max = generate_synthetic_dcm()
                    
                    # Show class distribution
                    classes, counts = np.unique(y_data, return_counts=True)
                    class_names = ["Normal", "Crack", "Wear"]
                    class_distribution = {class_names[c]: count for c, count in zip(classes, counts)}
                    
                    st.sidebar.success("✅ Data regenerated successfully!")
                    st.sidebar.write("Class distribution:", class_distribution)
                    
                    # Reset session state
                    if 'model' in st.session_state:
                        del st.session_state['model']
                    if 'metrics' in st.session_state:
                        del st.session_state['metrics']
                except Exception as e:
                    st.sidebar.error(f"Error generating data: {e}")
    
    # Training options
    st.sidebar.subheader("Training Options")
    epochs = st.sidebar.slider("Epochs", 10, 50, 25, help="Control training iterations")
    batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32], index=1)
    
    # Load or generate data
    try:
        # Check if data exists
        data_path = Path("mri_coil_data/data.npz")
        if not data_path.exists():
            st.info("No data found. Please generate synthetic data first.")
            X_data, y_data, filenames, X_min, X_max = generate_synthetic_dcm()
        else:
            data = np.load(data_path)
            X_data, y_data = data['X'], data['y']
            filenames = data['filenames']
            X_min, X_max = data['X_min'], data['X_max']
            
        # Split data
        indices = np.arange(len(X_data))
        np.random.shuffle(indices)
        X_data = X_data[indices]
        y_data = y_data[indices]
        filenames = np.array(filenames)[indices]
        
        train_size = int(0.8 * len(X_data))
        X_train, X_test = X_data[:train_size], X_data[train_size:]
        y_train, y_test = y_data[:train_size], y_data[train_size:]
        
        # Display class distribution
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        class_names = ["Normal", "Crack", "Wear"]
        train_distribution = {class_names[c]: count for c, count in zip(train_classes, train_counts)}
        
        test_classes, test_counts = np.unique(y_test, return_counts=True)
        test_distribution = {class_names[c]: count for c, count in zip(test_classes, test_counts)}
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Data")
            st.write(f"Total samples: {len(X_train)}")
            st.write("Class distribution:", train_distribution)
        
        with col2:
            st.subheader("Testing Data")
            st.write(f"Total samples: {len(X_test)}")
            st.write("Class distribution:", test_distribution)
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
        
    # Metadata for normalization
    metadata = load_metadata()
    
    # Train model button
    # In your main() function, find this code:
    train_button = st.sidebar.button("Train Balanced Model")
    if train_button:
        with st.spinner(f"Training balanced CNN for {epochs} epochs with class weights..."):
            try:
                # Train a balanced model with special handling for class imbalance
                model, history = train_balanced_model(
                    X_train, y_train, X_test, y_test, 
                    epochs=epochs, batch_size=batch_size
                )
                
                # Evaluate with confusion matrix
                metrics = evaluate_model_with_confusion_matrix(model, X_test, y_test)
                
                # Save model and metrics
                model.save("mri_coil_fault_detector.h5")
                st.session_state['model'] = model
                st.session_state['metrics'] = metrics
                st.session_state['history'] = history.history
                
                # Display success message
                st.success(f"Model trained! Test Accuracy: {metrics['test_acc']:.4f}")
                
                # Show confusion matrix
                cm_fig = display_confusion_matrix(metrics['confusion_matrix'])
                st.pyplot(cm_fig)
                
                # Display training curves
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Loss curve
                ax1.plot(history.history['loss'], label='Training Loss')
                ax1.plot(history.history['val_loss'], label='Validation Loss')
                ax1.set_title("Loss Over Epochs")
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss")
                ax1.legend()
                
                # Accuracy curve
                ax2.plot(history.history['accuracy'], label='Training Accuracy')
                ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax2.set_title("Accuracy Over Epochs")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Accuracy")
                ax2.legend()
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Training error: {str(e)}")
                logging.error(f"Training error: {str(e)}")

    # And ADD (don't replace) this code for a new button:
    brain_viz_button = st.sidebar.button("Train with Brain Visualization")
    if brain_viz_button:
        try:
            # Use the fixed function that preserves the original training approach
            model, history = train_balanced_model_with_visualization(
                X_train, y_train, X_test, y_test, 
                epochs=epochs, 
                batch_size=batch_size
            )
            
            # Save model and metrics
            model.save("mri_coil_fault_detector.h5")
            st.session_state['model'] = model
            st.session_state['history'] = history.history
            
            # Evaluate
            metrics = evaluate_model_with_confusion_matrix(model, X_test, y_test)
            st.session_state['metrics'] = metrics
            
            # Display success message
            st.success(f"Model trained! Test Accuracy: {metrics['test_acc']:.4f}")
            
            # Display confusion matrix
            cm_fig = display_confusion_matrix(metrics['confusion_matrix'])
            st.pyplot(cm_fig)
            
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            logging.error(f"Training error: {str(e)}")

    # train_button = st.sidebar.button("Train Balanced Model")
    # if train_button:
    #     with st.spinner(f"Training balanced CNN for {epochs} epochs with class weights..."):
    #         try:
    #             # Train a balanced model with special handling for class imbalance
    #             model, history = train_balanced_model(
    #                 X_train, y_train, X_test, y_test, 
    #                 epochs=epochs, batch_size=batch_size
    #             )
                
    #             # Evaluate with confusion matrix
    #             metrics = evaluate_model_with_confusion_matrix(model, X_test, y_test)
                
    #             # Save model and metrics
    #             model.save("mri_coil_fault_detector.h5")
    #             st.session_state['model'] = model
    #             st.session_state['metrics'] = metrics
    #             st.session_state['history'] = history.history
                
    #             # Display success message
    #             st.success(f"Model trained! Test Accuracy: {metrics['test_acc']:.4f}")
                
    #             # Show confusion matrix
    #             cm_fig = display_confusion_matrix(metrics['confusion_matrix'])
    #             st.pyplot(cm_fig)
                
    #             # Display training curves
    #             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
    #             # Loss curve
    #             ax1.plot(history.history['loss'], label='Training Loss')
    #             ax1.plot(history.history['val_loss'], label='Validation Loss')
    #             ax1.set_title("Loss Over Epochs")
    #             ax1.set_xlabel("Epoch")
    #             ax1.set_ylabel("Loss")
    #             ax1.legend()
                
    #             # Accuracy curve
    #             ax2.plot(history.history['accuracy'], label='Training Accuracy')
    #             ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    #             ax2.set_title("Accuracy Over Epochs")
    #             ax2.set_xlabel("Epoch")
    #             ax2.set_ylabel("Accuracy")
    #             ax2.legend()
                
    #             st.pyplot(fig)
                
    #         except Exception as e:
    #             st.error(f"Training error: {str(e)}")
    #             logging.error(f"Training error: {str(e)}")
    
    # Load existing model if available
    if 'model' not in st.session_state and os.path.exists("mri_coil_fault_detector.h5"):
        try:
            model = tf.keras.models.load_model("mri_coil_fault_detector.h5")
            st.session_state['model'] = model
            
            # Evaluate loaded model
            metrics = evaluate_model_with_confusion_matrix(model, X_test, y_test)
            st.session_state['metrics'] = metrics
            
            st.info("Loaded pre-trained model")
            
            # Show confusion matrix
            cm_fig = display_confusion_matrix(metrics['confusion_matrix'])
            st.pyplot(cm_fig)
            
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            logging.error(f"Model load error: {str(e)}")
    
    # Display model metrics if available
    if 'metrics' in st.session_state:
        metrics = st.session_state['metrics']
        
        st.subheader("Model Evaluation")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Accuracy", f"{metrics['test_acc']:.2%}")
        with col2:
            st.metric("Test Loss", f"{metrics['test_loss']:.4f}")
        
        # Display per-class metrics
        st.subheader("Per-Class Performance")
        
        class_data = []
        for class_name, class_metrics in metrics["class_metrics"].items():
            class_data.append({
                "Class": class_name,
                "Precision": f"{class_metrics['precision']:.2%}",
                "Recall": f"{class_metrics['recall']:.2%}",
                "F1 Score": f"{class_metrics['f1']:.2%}",
                "Count": class_metrics['count']
            })
        
        st.table(class_data)
    
    # Fault detection section
    st.header("Fault Detection")
    
    # Image source selection
    image_source = st.radio("Image Source:", ["Upload DICOM", "Test with Normal", "Test with Crack", "Test with Wear"])
    
    image = None
    if image_source == "Upload DICOM":
        uploaded_file = st.file_uploader("Upload a coil image (.dcm)", type=["dcm"])
        if uploaded_file:
            st.write("File uploaded successfully")
            try:
                image = load_dcm(uploaded_file, metadata)
                if image is None:
                    st.error("Failed to load DICOM file")
                elif image.shape[0] != 32 or image.shape[1] != 32:
                    st.error("Image must be 32x32 pixels")
            except Exception as e:
                st.error(f"Error processing file: {e}")
                logging.error(f"File processing error: {str(e)}")
    else:
        # Generate test sample based on selection
        img_size = 32
        sample_type = image_source.split()[-1].lower()
        
        if sample_type == "normal":
            # Generate normal test image
            image_raw = np.ones((img_size, img_size), dtype=np.float32) * NORMAL_INTENSITY
            image_raw += np.random.normal(0, 5, (img_size, img_size))
            image_raw = np.clip(image_raw, NORMAL_INTENSITY-10, NORMAL_INTENSITY+10)
        elif sample_type == "crack":
            # Generate crack test image
            image_raw = np.ones((img_size, img_size), dtype=np.float32) * BACKGROUND_INTENSITY
            crack_width = 3
            for j in range(img_size):
                if j < img_size - crack_width:
                    for w in range(crack_width):
                        image_raw[j, j+w] = CRACK_INTENSITY
        elif sample_type == "wear":
            # Generate wear test image
            image_raw = np.ones((img_size, img_size), dtype=np.float32) * BACKGROUND_INTENSITY
            num_spots = 25
            for _ in range(num_spots):
                x = np.random.randint(0, img_size)
                y = np.random.randint(0, img_size)
                spot_size = np.random.randint(3, 5)
                for dx in range(-spot_size, spot_size+1):
                    for dy in range(-spot_size, spot_size+1):
                        if 0 <= x+dx < img_size and 0 <= y+dy < img_size:
                            distance = np.sqrt(dx*dx + dy*dy)
                            if distance <= spot_size:
                                image_raw[x+dx, y+dy] = WEAR_INTENSITY
        
        # Normalize the test image using metadata
        image = (image_raw - metadata["X_min"]) / (metadata["X_max"] - metadata["X_min"])
        image = np.clip(image, 0.0, 1.0)
        image = image[..., np.newaxis]  # Add channel dimension
        st.write(f"Generated test {sample_type} sample")
    
    # If image is loaded, display and analyze
    if image is not None:
        # Layout columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display image
            st.subheader("Input Image")
            st.image(image.squeeze(), caption="Coil Image", use_column_width=True)
            
            # Extract and display image features
            features = get_image_features(image)
            st.subheader("Image Features")
            
            # Show key metrics that help distinguish classes
            metrics_to_show = {
                "Mean Intensity": features["mean"],
                "Standard Deviation": features["std"],
                "Diagonal Max": features["diagonal_max"],
                "High Intensity %": features["high_intensity_ratio"] * 100,
                "Entropy": features["entropy"]
            }
            
            for key, value in metrics_to_show.items():
                st.metric(key, f"{value:.4f}")
            
            # Show histogram of pixel intensities
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(image.flatten(), bins=30, range=(0, 1))
            ax.set_title("Pixel Intensity Distribution")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        
        with col2:
            # Analyze image if model is available
            if 'model' in st.session_state:
                model = st.session_state['model']
                
                # Get prediction directly from model
                status, prob, pred_probs = predict_fault_direct(model, image)
                
                # Define recommended actions
                actions = {
                    "Normal": "No action needed",
                    "Crack Detected": "Replace MRI coil immediately due to crack",
                    "Wear Detected": "Schedule maintenance for MRI coil wear"
                }
                action = actions.get(status, "Unknown fault - inspect manually")
                
                # Display prediction results
                st.subheader("Fault Analysis")
                
                # Create metrics for prediction results
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    st.metric("Prediction", status)
                with col_pred2:
                    st.metric("Confidence", f"{prob:.2%}")
                
                st.info(f"**Recommended Action:** {action}")
                
                # Create bar chart for class probabilities
                fig, ax = plt.subplots(figsize=(8, 4))
                classes = ["Normal", "Crack", "Wear"]
                colors = ['green', 'red', 'orange']
                ax.bar(classes, pred_probs, color=colors)
                ax.set_title("Fault Probability Distribution")
                ax.set_ylim(0, 1)
                for i, p in enumerate(pred_probs):
                    ax.text(i, p + 0.05, f"{p:.2%}", ha='center')
                st.pyplot(fig)
                
                # Manual verification
                st.subheader("Verification")
                if st.button("Verify Manually"):
                    manual_result = verify_manually(image)
                    st.write(f"**Manual Verification Result:** {manual_result}")
                    
                    # Check if manual verification matches prediction
                    if "Crack" in manual_result and "Crack" in status:
                        st.success("✓ Manual verification confirms prediction")
                    elif "Wear" in manual_result and "Wear" in status:
                        st.success("✓ Manual verification confirms prediction")
                    elif "Normal" in manual_result and status == "Normal":
                        st.success("✓ Manual verification confirms prediction")
                    else:
                        st.warning("⚠ Manual verification differs from prediction")
                
                # Log repair
                if st.button("Log Repair"):
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"{timestamp} | Status: {status} | Prob: {prob:.4f} | Action: {action}"
                    try:
                        # Make sure directory exists
                        os.makedirs(os.path.dirname("logs/"), exist_ok=True)
                        
                        with open("logs/mri_repair_log.txt", "a") as f:
                            f.write(log_entry + "\n")
                        logging.info(f"Logged: {log_entry}")
                        st.success(f"Logged: {log_entry}")
                    except Exception as e:
                        st.error(f"Logging error: {e}")
                
                # Visualization of neural network
                # Replace the visualization code block in the main function with this:

# Visualization of neural network
        st.subheader("Neural Network Visualization")

        # Get actual layer names from the model to populate the dropdown
        try:
            available_layers = [layer.name for layer in model.layers]
            logging.info(f"Available layers for visualization: {available_layers}")
        except Exception as e:
            available_layers = ['conv1', 'pool1', 'conv2', 'pool2', 'flatten', 'dense1', 'output']
            logging.error(f"Could not get layer names: {str(e)}")

        selected_layers = st.multiselect(
            "Select Layers to Visualize", 
            available_layers,
            default=[available_layers[0]] if available_layers else []
        )

        if selected_layers and st.button("Visualize Layers"):
            try:
                # Use the extremely robust visualization function
                with st.spinner("Generating layer visualizations..."):
                    figures = visualize_layers_robust(model, image, selected_layers)
                    
                    # Display the figures
                    if figures:
                        for fig in figures:
                            st.pyplot(fig)
                    else:
                        st.warning("No visualizations could be generated.")
                        st.info("Available layers in model: " + ", ".join(available_layers))
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
                st.info("Try selecting different layers or check if the model is properly loaded.")

            else:
                st.warning("Please train a model or load a pre-trained model first")
    
    # Footer
    st.markdown("---")
    st.write("Developed by Jim Gitonga | Powered by POWER_OF_TENSOR | © 2025")

if __name__ == "__main__":
    main()

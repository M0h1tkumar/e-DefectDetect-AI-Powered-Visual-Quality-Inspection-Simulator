import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import time # For performance monitoring

# For model performance tracking - RESTORING IMPORTS
import os
import glob
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# --- CONFIGURATION ---
# !!! IMPORTANT: CUSTOMIZE THESE PATHS AND PARAMETERS FOR YOUR MODEL !!!
MODEL_PATH = 'best_vgg19_model.h5' # <--- CHANGE THIS if your model file name is different or in a subfolder
LABELS_PATH = 'labels.txt' # <--- CHANGE THIS if your labels file name is different or in a subfolder
IMAGE_SIZE = (75, 75) # <--- CHANGE THIS to match your model's input image size (height, width)
# The EXACT name of the layer in your model that is the VGG19 base or the last convolutional layer
# For example, if your model's summary shows 'vgg19', 'resnet50', 'efficientnetb0' etc.
# This is crucial for Grad-CAM.
LAST_CONV_LAYER_NAME_TENSORFLOW = 'vgg19' # <--- VERIFY/CHANGE THIS NAME

# Directory for validation images for performance tracking - RESTORING
# Create subfolders inside this directory, with each subfolder named after a class label
# E.g., validation_data/crazing, validation_data/inclusion
VALIDATION_IMAGES_DIR ='validation_data' # <--- IMPORTANT: SET THIS PATH (e.g., 'validation_data' if in same dir)

# --- Load Model Function ---
@st.cache_resource
def load_model():
    start_time = time.time()
    st.info("Loading TensorFlow/Keras model... Please wait.")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        end_time = time.time()
        st.success(f"TensorFlow/Keras model loaded successfully! (Took {end_time - start_time:.2f} seconds)")
        # model.summary() # Uncomment for debugging during development, remove for production
        return model
    except Exception as e:
        st.error(f"Error loading TensorFlow/Keras model from {MODEL_PATH}: {e}")
        st.error("Please check the MODEL_PATH and ensure the model file is accessible.")
        return None

# --- Load Labels Function ---
@st.cache_data
def load_labels():
    """Loads class labels from a text file."""
    try:
        with open(LABELS_PATH, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]

        if not labels:
            st.warning(f"Warning: Labels file '{LABELS_PATH}' is empty. Please add class labels, one per line.")
            return []
        st.success("Class labels loaded successfully!")
        return labels
    except FileNotFoundError:
        st.error(f"Error: Labels file '{LABELS_PATH}' not found.")
        st.info("Please create a 'labels.txt' file in the same directory as your 'app.py' and 'model' folder.")
        st.info("The 'labels.txt' file should contain one class label per line, in the exact order your model was trained on.")
        st.info("Example 'labels.txt':\n```\nNo Defect\nScratch\nDent\nColor Mismatch\nCracks\n```")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading labels: {e}")
        st.info("Please ensure 'labels.txt' is a plain text file and correctly formatted.")
        return []

# --- Preprocessing Function ---
def preprocess_image(image_input):
    """
    Applies the same preprocessing transformations as used during model training.
    Assumes image_input is a PIL Image object.
    """
    if image_input.mode != "RGB":
        image_input = image_input.convert("RGB")

    img_array = image.img_to_array(image_input.resize(IMAGE_SIZE))
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    # IMPORTANT: Apply the SAME normalization used during TRAINING
    # For VGG19 typically, it might be preprocess_input from tf.keras.applications.vgg19
    # If your model was trained with /255.0, keep it.
    img_array = img_array / 255.0 # Example normalization

    return img_array

# --- Prediction Function ---
def predict(model, image_tensor, labels):
    """Makes a prediction using the loaded model."""
    start_time = time.time()
    predictions = model.predict(image_tensor, verbose=0) # Set verbose=0 to suppress console output
    end_time = time.time()
    st.info(f"Prediction took {end_time - start_time:.4f} seconds.")

    predicted_idx = np.argmax(predictions[0])
    predicted_class = labels[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100

    top_n = min(5, len(labels)) # Get top 5 or fewer if fewer classes
    top_indices = predictions[0].argsort()[-top_n:][::-1]
    top_predictions = [(labels[idx], predictions[0][idx] * 100) for idx in top_indices]

    return predicted_class, confidence, top_predictions, predicted_idx, predictions[0] # Return full prediction probabilities

# --- Grad-CAM Implementation for TensorFlow/Keras ---
@st.cache_data(show_spinner=False) # Cache Grad-CAM results for same inputs
def get_grad_cam_tensorflow(model, input_image_array, target_class_idx, last_conv_layer_name):
    """
    Generates a Grad-CAM heatmap for TensorFlow/Keras models by explicitly constructing
    feature extractor and classifier sub-models. This is robust against graph
    issues in loaded Sequential models with nested Functional models.
    """
    # st.info("Generating Grad-CAM (cached)...") # For debugging cache
    try:
        target_top_level_layer = model.get_layer(last_conv_layer_name)
        actual_last_conv_layer_obj_name = None

        if isinstance(target_top_level_layer, tf.keras.Model) and len(target_top_level_layer.layers) > 0:
            found_inner_conv = False
            for layer_in_submodel in reversed(target_top_level_layer.layers):
                if isinstance(layer_in_submodel, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D, tf.keras.layers.DepthwiseConv2D)):
                    actual_last_conv_layer_obj_name = layer_in_submodel.name
                    found_inner_conv = True
                    break
            if not found_inner_conv:
                st.error(f"Error: No Conv2D layer found inside the sub-model '{last_conv_layer_name}'.")
                return None
        else:
            if isinstance(target_top_level_layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D, tf.keras.layers.DepthwiseConv2D)):
                actual_last_conv_layer_obj_name = target_top_level_layer.name
            else:
                st.error(f"Error: Layer '{last_conv_layer_name}' is not a convolutional layer or a sub-model containing one.")
                return None

        if actual_last_conv_layer_obj_name is None:
            st.error(f"Error: Could not identify the last convolutional layer object using name '{last_conv_layer_name}'.")
            return None

        # Build the feature extractor model
        feature_extractor_input = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        x = feature_extractor_input
        feature_outputs = None

        for layer in model.layers:
            if layer == target_top_level_layer:
                x = layer(x)
                feature_outputs = x
                break
            else:
                x = layer(x)

        if feature_outputs is None:
            raise ValueError(
                f"Internal error: Failed to trace path to layer '{actual_last_conv_layer_obj_name}' "
                f"because its containing top-level layer ('{last_conv_layer_name}') was not encountered "
                f"in the main model's layer list for feature extraction. "
                f"Model layers found: {[l.name for l in model.layers]}"
            )

        feature_extractor = Model(inputs=feature_extractor_input, outputs=feature_outputs, name="feature_extractor_model")

        # Build the classifier model
        classifier_input_tensor = Input(shape=feature_outputs.shape[1:])
        y = classifier_input_tensor

        found_start_of_classifier = False
        for layer in model.layers:
            if layer == target_top_level_layer:
                found_start_of_classifier = True
                continue # Skip the feature extractor layer itself, we're starting after it
            if found_start_of_classifier:
                y = layer(y)

        classifier = Model(inputs=classifier_input_tensor, outputs=y, name="classifier_model")

        # Compute gradients
        with tf.GradientTape() as tape:
            last_conv_layer_activations = feature_extractor(input_image_array)
            tape.watch(last_conv_layer_activations)
            preds = classifier(last_conv_layer_activations)
            target_score = preds[:, target_class_idx] # Score for the predicted class

        grads = tape.gradient(target_score, last_conv_layer_activations)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # Global average pooling of gradients

        # Multiply each channel in the feature map by the importance (pooled_grads) for that channel
        heatmap = last_conv_layer_activations[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap) # Remove single-dimensional entries
        heatmap = tf.maximum(heatmap, 0) # Apply ReLU to heatmap to get only positive influence

        max_val = tf.reduce_max(heatmap)
        if max_val == 0:
            return None
        heatmap = heatmap / max_val # Normalize heatmap to [0, 1]

        return heatmap.numpy()

    except ValueError as e:
        st.error(f"TensorFlow Grad-CAM ValueError: {e}")
        st.exception(e)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during Grad-CAM generation: {e}")
        st.exception(e)
        return None

# --- display_grad_cam Function ---
def display_grad_cam(original_image_pil, heatmap, alpha=0.4):
    if heatmap is None:
        return None

    # Resize original image to match model input size for consistent overlay
    img_for_overlay = np.array(original_image_pil.resize(IMAGE_SIZE))
    img_for_overlay_bgr = cv2.cvtColor(img_for_overlay, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV

    # Resize heatmap to original image dimensions for overlay
    heatmap_resized = cv2.resize(heatmap, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized) # Scale to 0-255
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET) # Apply JET colormap

    # Overlay the heatmap on the original image
    superimposed_img_bgr = cv2.addWeighted(heatmap_colored, alpha, img_for_overlay_bgr, 1 - alpha, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB) # Convert back to RGB for PIL/Streamlit

    return Image.fromarray(superimposed_img_rgb)


# --- Model Performance Tracking Functions - RESTORING ---

@st.cache_data(show_spinner=True)
def get_validation_predictions_and_labels(_model, _labels, validation_dir, image_size):
    """
    Loads validation images, makes predictions, and returns true and predicted labels,
    along with raw prediction probabilities and image paths.
    """
    if not os.path.isdir(validation_dir):
        st.error(f"Validation directory '{validation_dir}' not found. Cannot perform performance tracking.")
        return [], [], [], []

    true_labels = []
    predicted_labels = []
    all_probabilities = []
    image_paths = []

    class_folders = sorted([d for d in os.listdir(validation_dir) if os.path.isdir(os.path.join(validation_dir, d))])
    if not class_folders:
        st.warning(f"No class subfolders found in '{validation_dir}'. Performance tracking might be incomplete.")
        return [], [], [], []

    # Map folder names to label indices
    label_to_idx = {label: idx for idx, label in enumerate(_labels)}

    # Filter class_folders to only include those that match labels in labels.txt
    filtered_class_folders = [f for f in class_folders if f in label_to_idx]
    if not all(folder in label_to_idx for folder in class_folders):
        st.warning(f"Some subfolders in '{validation_dir}' do not match defined labels in labels.txt. "
                   f"Subfolders found: {class_folders}, Labels in labels.txt: {_labels}. "
                   f"Only processing images from matching folders: {filtered_class_folders}.")
    class_folders = filtered_class_folders # Use filtered list

    progress_text = "Processing validation images... Please wait."
    my_bar = st.progress(0, text=progress_text)
    total_files = sum([len(glob.glob(os.path.join(validation_dir, cf, '*'))) for cf in class_folders])
    processed_files = 0

    if total_files == 0:
        st.warning(f"No images found in '{validation_dir}' or its subfolders. Cannot perform performance tracking.")
        return [], [], [], []

    for class_folder in class_folders:
        true_label_idx = label_to_idx.get(class_folder)
        # This check is now redundant due to filtering above, but harmless.
        if true_label_idx is None:
            continue

        folder_path = os.path.join(validation_dir, class_folder)
        for img_file in glob.glob(os.path.join(folder_path, '*.jpg')) + \
                         glob.glob(os.path.join(folder_path, '*.jpeg')) + \
                         glob.glob(os.path.join(folder_path, '*.png')):
            try:
                img = Image.open(img_file)
                processed_img = preprocess_image(img)
                predictions = _model.predict(processed_img, verbose=0)[0] # Get raw probabilities
                predicted_idx = np.argmax(predictions)

                true_labels.append(true_label_idx)
                predicted_labels.append(predicted_idx)
                all_probabilities.append(predictions)
                image_paths.append(img_file)

                processed_files += 1
                my_bar.progress(processed_files / total_files, text=progress_text)
            except Exception as e:
                st.warning(f"Could not process image {img_file}: {e}")
                continue
    my_bar.empty()
    st.success(f"Processed {processed_files} validation images.")
    return np.array(true_labels), np.array(predicted_labels), np.array(all_probabilities), image_paths

def plot_confusion_matrix(cm, class_names):
    """Plots the confusion matrix using Seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    st.pyplot(plt)
    plt.close() # Close plot to prevent memory issues

def calculate_top_k_errors(true_indices, probabilities, k, labels, image_paths):
    """
    Calculates top-k accuracy and identifies examples where the true label
    is not in the top K predictions.
    """
    if len(true_indices) == 0:
        return 0.0, []

    # Use sklearn's top_k_accuracy_score for accuracy calculation
    top_k_accuracy = top_k_accuracy_score(true_indices, probabilities, k=k, labels=np.arange(len(labels)))

    top_k_errors_list = []

    for i in range(len(true_indices)):
        true_idx = true_indices[i]
        probs = probabilities[i]
        img_path = image_paths[i]

        # Get top K predicted indices based on probabilities
        top_k_indices = np.argsort(probs)[::-1][:k] # Sort descending and take top K

        if true_idx not in top_k_indices:
            # This is a top-K error, record details
            top_5_predictions_for_error = []
            # Get top 5 predictions for this specific error to show context
            top_5_indices_all = np.argsort(probs)[::-1][:min(5, len(labels))]
            for idx in top_5_indices_all:
                top_5_predictions_for_error.append(f"{labels[idx]} ({probs[idx]*100:.2f}%)")

            top_k_errors_list.append({
                'image_path': img_path,
                'true_label': labels[true_idx],
                'predicted_top1': labels[np.argmax(probs)],
                'top_5_predictions': top_5_predictions_for_error
            })

    return top_k_accuracy, top_k_errors_list

# --- Streamlit UI Layout ---
st.set_page_config(
    page_title="Image Classification & Explainability App (TensorFlow)",
    page_icon="📸",
    layout="centered" # or "wide" for more space
)

st.title("Image Classification App with Grad-CAM (TensorFlow/Keras)")
st.markdown("Upload an image, get a prediction, and see *why* the model made that prediction!")

st.sidebar.header("Configuration & Info")
st.sidebar.info("This app is configured for **TensorFlow/Keras** models.")
st.sidebar.markdown(f"**Current local time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")

# --- Load model and labels ---
model = load_model()
labels = load_labels()

if model is None or not labels:
    st.error("Application cannot run without a loaded model and labels. Please resolve the issues above.")
else:
    # --- Single Image Prediction Section ---
    st.header("Single Image Prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image, caption='Uploaded Image.', use_container_width=True)
        st.write("")

        st.subheader("Prediction Results:")

        try:
            # We need the full probabilities for Grad-CAM's target_class_idx and for performance monitoring
            processed_image_tensor = preprocess_image(original_image)
            predicted_class, confidence, top_predictions, predicted_idx, full_probabilities = predict(model, processed_image_tensor, labels)


            st.success(f"**Prediction: {predicted_class}** (Confidence: {confidence:.2f}%)")

            st.write("---")
            st.subheader("Top Predictions:")
            col1, col2 = st.columns(2)
            for i, (class_name, prob) in enumerate(top_predictions):
                if i % 2 == 0:
                    col1.write(f"- **{class_name}**: {prob:.2f}%")
                else:
                    col2.write(f"- **{class_name}**: {prob:.2f}%")

            st.write("---")
            st.subheader("Explainability (Grad-CAM):")
            show_grad_cam = st.checkbox(
                "Show Grad-CAM Heatmap",
                value=True, # Default to checked for easy viewing
                help="Visualize the areas of the image that most influenced the model's prediction for the predicted class."
            )

            if show_grad_cam:
                with st.spinner("Generating Grad-CAM heatmap..."):
                    heatmap = get_grad_cam_tensorflow(
                        model, processed_image_tensor, predicted_idx, LAST_CONV_LAYER_NAME_TENSORFLOW
                    )

                    if heatmap is not None:
                        alpha_slider = st.slider(
                            "Heatmap Intensity (Alpha)",
                            0.0, 1.0, 0.5, 0.05,
                            help="Adjust the transparency of the heatmap overlay.",
                            key="grad_cam_alpha_slider" # Unique key for Streamlit sliders
                        )
                        grad_cam_image = display_grad_cam(original_image, heatmap, alpha=alpha_slider)
                        st.image(grad_cam_image, caption=f'Grad-CAM for "{predicted_class}"', use_container_width=True)
                    else:
                        st.warning("Could not generate Grad-CAM heatmap.")
                        st.info("Check console for potential errors during heatmap generation or if activation was too low for the predicted class.")

        except Exception as e:
            st.error(f"An unexpected error occurred during prediction or Grad-CAM generation: {e}")
            st.exception(e) # Displays full traceback for debugging
            st.info("Ensure your model is compatible and preprocessing steps match your training.")

    st.markdown("---")

    # --- Model Performance Tracking Section - RESTORING ---
    st.header("Model Performance Tracking")
    st.info("This section evaluates the model on a dedicated validation dataset. Please ensure your validation images are correctly set up in the specified `VALIDATION_IMAGES_DIR`.")

    if st.button("Evaluate Model Performance", key="evaluate_performance_btn"):
        if not os.path.exists(VALIDATION_IMAGES_DIR):
            st.error(f"Validation data directory '{VALIDATION_IMAGES_DIR}' not found. Please create it and add images in class subfolders.")
        else:
            with st.spinner("Evaluating model on validation data... This may take a while for large datasets."):
                true_labels_val, predicted_labels_val, all_probabilities_val, image_paths_val = \
                    get_validation_predictions_and_labels(model, labels, VALIDATION_IMAGES_DIR, IMAGE_SIZE)

                if len(true_labels_val) > 0:
                    st.subheader("Overall Accuracy")
                    overall_accuracy = accuracy_score(true_labels_val, predicted_labels_val)
                    st.success(f"**Overall Accuracy: {overall_accuracy:.2f}**") # Display as a float

                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(true_labels_val, predicted_labels_val)
                    plot_confusion_matrix(cm, labels)
                    st.write("The confusion matrix shows the counts of true positives, true negatives, false positives, and false negatives.")
                    st.markdown("""
                    - **Rows:** True Labels
                    - **Columns:** Predicted Labels
                    - **Diagonal:** Correct predictions
                    - **Off-diagonal:** Misclassifications (errors)
                    """)

                    st.subheader("Top-K Accuracy and Errors")
                    default_k = min(5, len(labels)) # Default K value, can't be more than number of classes
                    k_value = st.slider("Select K for Top-K Accuracy", 1, len(labels), default_k, help="Number of top predictions to consider for Top-K accuracy.")

                    top_k_accuracy, top_k_errs = calculate_top_k_errors(
                        true_labels_val, all_probabilities_val, k_value, labels, image_paths_val
                    )
                    st.info(f"**Top-{k_value} Accuracy: {top_k_accuracy:.2f}**") # Display as a float
                    st.warning(f"Total Top-{k_value} Errors: {len(top_k_errs)}")

                    if top_k_errs:
                        st.markdown(f"**Examples of Top-{k_value} Errors:**")
                        # Display up to 5 examples to keep UI from getting too long
                        display_limit = 5
                        for i, error_info in enumerate(top_k_errs[:display_limit]):
                            st.write(f"--- **Error {i+1}** ---")
                            st.write(f"**Image:** `{os.path.basename(error_info['image_path'])}` (from `{os.path.basename(os.path.dirname(error_info['image_path']))}` folder)")
                            st.write(f"**True Label:** `{error_info['true_label']}`")
                            st.write(f"**Predicted Top 1:** `{error_info['predicted_top1']}`")
                            st.write(f"**Top 5 Predictions:** {', '.join(error_info['top_5_predictions'])}")
                            try:
                                error_image = Image.open(error_info['image_path'])
                                st.image(error_image, caption=f"Error Image (True: {error_info['true_label']})", width=200)
                            except Exception as img_e:
                                st.warning(f"Could not load error image {error_info['image_path']}: {img_e}")
                        if len(top_k_errs) > display_limit:
                            st.info(f"Showing {display_limit} of {len(top_k_errs)} Top-{k_value} errors. For a full list of errors, you might need to inspect the data outside the app.")
                    else:
                        st.success(f"No Top-{k_value} errors found (meaning all true labels were within the top {k_value} predictions). Excellent performance!")
                else:
                    st.warning("No validation images were processed. Ensure your `VALIDATION_IMAGES_DIR` is correct and contains images in class subfolders with names matching `labels.txt`.")


st.markdown("---")
st.markdown("Developed by Mohit Kumar, Jamshedpur, Jharkhand, India.")

from flask import Flask, render_template, request, url_for, send_from_directory
import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import faiss
import json

app = Flask(__name__)

# --- Configuration ---
DATA_DIR = os.path.join(app.root_path, 'data')
FEATURES_FILE = os.path.join(DATA_DIR, 'fashion_features.npy')
LABELS_FILE = os.path.join(DATA_DIR, 'fashion_labels.npy')
INDEX_FILE = os.path.join(DATA_DIR, 'fashion_faiss_index.bin')
METADATA_FILE = os.path.join(DATA_DIR, 'fashion_metadata.json')

# ### IMPORTANT CHANGE: Set to 60000 to process the full dataset
DATASET_FULL_SIZE = 60000  # Process the full 60,000 images for features

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'temp_images'), exist_ok=True)

# Device for PyTorch (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Global Variables for ML Assets ---
model = None
all_features = None
all_labels = None
faiss_index = None
# This is the full Fashion-MNIST training dataset for display purposes
full_fashion_dataset_for_display = None
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Data Transformations for Images ---
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
display_transform = transforms.Compose([transforms.ToTensor()])  # For displaying original images (no normalization)


# --- ML Model Loading and Feature Extraction ---
def load_or_generate_ml_assets():
    """
    Loads pre-computed image features and FAISS index if they exist.
    Otherwise, downloads Fashion-MNIST, extracts features using ResNet,
    builds a FAISS index, and saves these assets.
    """
    global model, all_features, all_labels, faiss_index, full_fashion_dataset_for_display

    # Load Fashion-MNIST full dataset first (for both feature extraction and display)
    full_fashion_dataset_for_display = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True,
                                                             transform=display_transform)
    # This dataset will be used for feature extraction. We'll ensure it's the full 60k.
    full_train_dataset_for_features = datasets.FashionMNIST(root=DATA_DIR, train=True, download=False,
                                                            transform=transform)

    # Try to load pre-computed assets from disk
    if (os.path.exists(FEATURES_FILE) and
            os.path.exists(LABELS_FILE) and
            os.path.exists(INDEX_FILE) and
            os.path.exists(METADATA_FILE)):  # Check for metadata file too
        print("Loading pre-computed features, labels, FAISS index, and metadata...")
        all_features = np.load(FEATURES_FILE)
        all_labels = np.load(LABELS_FILE)
        faiss_index = faiss.read_index(INDEX_FILE)

        print(f"Type of faiss_index after loading: {type(faiss_index)}")
        print(f"FAISS index total vectors after loading: {faiss_index.ntotal}")

        # Ensure the ResNet model is also loaded for potential new image processing (e.g., user uploads)
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Identity()  # Remove the final classification layer
        model.eval()  # Set model to evaluation mode
        model.to(DEVICE)
        print("ML assets loaded.")
        return

    # If pre-computed assets don't exist, generate them (using the full dataset)
    print(f"Pre-computed ML assets not found. Generating them now (processing {DATASET_FULL_SIZE} images)...")

    # When DATASET_FULL_SIZE is 60000, we process the full dataset directly.
    # No subsetting needed here.

    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(DEVICE)

    all_features_list = []
    all_labels_list = []

    # Use the full dataset with the DataLoader
    dataloader = DataLoader(full_train_dataset_for_features, batch_size=64, shuffle=False, num_workers=0)

    print("Extracting features from Fashion-MNIST (full 60k) images...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(DEVICE)
            features = model(images)
            all_features_list.append(features.cpu().numpy())
            all_labels_list.append(labels.cpu().numpy())
            if (i + 1) % 100 == 0:  # Print progress more frequently for 60k
                print(f"Processed {(i + 1) * dataloader.batch_size} images...")

    all_features = np.vstack(all_features_list)
    all_labels = np.concatenate(all_labels_list)
    print(f"Total features extracted: {all_features.shape}")

    dimension = all_features.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(all_features)

    print(f"Type of faiss_index after creation: {type(faiss_index)}")
    print(f"FAISS index built with {faiss_index.ntotal} vectors.")

    np.save(FEATURES_FILE, all_features)
    np.save(LABELS_FILE, all_labels)
    faiss.write_index(faiss_index, INDEX_FILE)

    # Store metadata for direct mapping (original Fashion-MNIST index -> class name)
    # Since we are processing the full dataset in order, the index in all_features
    # directly corresponds to the original Fashion-MNIST dataset index.
    metadata = [
        {'original_fm_idx': i, 'label_id': all_labels[i].item(), 'class_name': class_names[all_labels[i].item()]}
        for i in range(len(all_labels))]
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

    print("ML assets generated and saved.")


# --- Flask Routes ---

print("Initializing ML assets...")
load_or_generate_ml_assets()
print("ML assets initialization complete.")

print(f"Global 'model' object at app startup: {type(model)}")
print(f"Global 'faiss_index' object at app startup: {type(faiss_index)}")


@app.route('/')
def index_page():
    global all_labels, full_fashion_dataset_for_display  # Ensure access to these globals

    # Hand-picked indices from the *original* Fashion-MNIST dataset that are generally clear
    # and match their categories well. These indices refer to the position in the *full* Fashion-MNIST dataset.
    GOOD_FASHION_MNIST_SAMPLE_INDICES = [
        0,  # T-shirt/top
        1,  # Trouser
        2,  # Pullover
        3,  # Dress
        4,  # Coat
        5,  # Sandal
        7,  # Sneaker
        10,  # Bag
        14,  # Ankle boot
        20  # Shirt
    ]

    sample_items = []

    # Clear any previous temporary images to ensure only current samples are shown
    temp_img_dir = os.path.join(app.root_path, 'static', 'temp_images')
    for f in os.listdir(temp_img_dir):
        os.remove(os.path.join(temp_img_dir, f))

    for original_fm_idx in GOOD_FASHION_MNIST_SAMPLE_INDICES:
        # Get the image and its true label from the Fashion-MNIST dataset using the original index
        # This part will work because full_fashion_dataset_for_display is the full 60k dataset.
        img, label_id = full_fashion_dataset_for_display[original_fm_idx]  # Use display_transform here

        # Get the class name from our predefined list
        class_name = class_names[label_id]

        img_pil = transforms.ToPILImage()(img)  # Convert PyTorch tensor to PIL Image

        # Use the original_fm_idx for the filename. This will also be the value passed to the backend.
        img_filename = f'sample_item_{original_fm_idx}.png'
        img_path = os.path.join(app.root_path, 'static', 'temp_images', img_filename)
        img_pil.save(img_path)

        sample_items.append({
            'original_idx': original_fm_idx,  # Pass the original Fashion-MNIST index
            'class_name': class_name,
            'image_url': url_for('serve_temp_image', filename=img_filename)
        })
    return render_template('index.html', sample_items=sample_items)


@app.route('/recommend', methods=['POST'])
def recommend():
    global faiss_index, all_features, all_labels, class_names, full_fashion_dataset_for_display

    if faiss_index is None or all_features is None or all_labels is None:
        return "Recommendation system assets not fully loaded. Please try again later.", 500

    # The index received from the form is the original Fashion-MNIST dataset index
    selected_original_fm_idx_str = request.form.get('selected_item_idx')

    if not selected_original_fm_idx_str:
        return "Please select an item to get recommendations.", 400

    try:
        selected_original_fm_idx = int(selected_original_fm_idx_str)
        # With DATASET_FULL_SIZE = 60000, selected_original_fm_idx directly corresponds to all_features index.
        if not (0 <= selected_original_fm_idx < len(all_features)):
            return "Selected item index out of bounds within the processed dataset.", 400
    except ValueError:
        return "Invalid item selected. Index must be an integer.", 400

    # Get the feature vector for the selected item directly from all_features
    query_feature = all_features[selected_original_fm_idx].reshape(1, -1)

    print(f"Attempting faiss_index.search. Type of faiss_index: {type(faiss_index)}")

    distances, nearest_indices = faiss_index.search(query_feature, 10 + 1)  # Get 11, filter 1 out

    # Filter out the query item itself (which will be the first result)
    # The indices returned by FAISS are now the original Fashion-MNIST indices
    recommended_original_fm_indices = [idx for idx in nearest_indices[0] if idx != selected_original_fm_idx][:10]

    recommendations = []

    # Clear previous temporary images before generating new ones for recommendations
    temp_img_dir = os.path.join(app.root_path, 'static', 'temp_images')
    for f in os.listdir(temp_img_dir):
        if f.startswith('rec_item_'):  # Only delete recommendation images, not samples
            os.remove(os.path.join(temp_img_dir, f))

    for original_fm_idx in recommended_original_fm_indices:
        # Get the image and its true label from the Fashion-MNIST dataset using the original index
        img, label_id = full_fashion_dataset_for_display[original_fm_idx]
        class_name = class_names[label_id]

        img_pil = transforms.ToPILImage()(img)

        img_filename = f'rec_item_{original_fm_idx}.png'
        img_path = os.path.join(app.root_path, 'static', 'temp_images', img_filename)
        img_pil.save(img_path)

        recommendations.append({
            'class_name': class_name,
            'image_url': url_for('serve_temp_image', filename=img_filename)
        })

    # Get details of the selected item for display on the recommendations page
    # Use the label directly from the original Fashion-MNIST dataset via its index
    selected_item_label_id = full_fashion_dataset_for_display.targets[selected_original_fm_idx].item()
    selected_item_class_name = class_names[selected_item_label_id]

    # Re-use the already saved sample image for the selected item
    selected_item_image_url = url_for('serve_temp_image', filename=f'sample_item_{selected_original_fm_idx}.png')

    return render_template('recommendations.html',
                           selected_item_class_name=selected_item_class_name,
                           selected_item_image_url=selected_item_image_url,
                           recommendations=recommendations)


@app.route('/static/temp_images/<path:filename>')
def serve_temp_image(filename):
    """
    Serves temporary image files stored in static/temp_images.
    """
    return send_from_directory(os.path.join(app.root_path, 'static', 'temp_images'), filename)


if __name__ == '__main__':
    # Ensure KMP_DUPLICATE_LIB_OK is set if you face OpenMP errors during local runs
    # This should be set in PyCharm's run configuration (Run -> Edit Configurations -> app -> Environment variables)
    app.run(debug=True)
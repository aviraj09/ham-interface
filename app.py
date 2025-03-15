import streamlit as st
import torch
from PIL import Image
import os
import uuid
import pandas as pd
from torchvision import transforms, models
import torch.nn as nn

# Define upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features

class CustomFC(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CustomFC, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model.fc = CustomFC(num_ftrs, 7)
model.load_state_dict(torch.load("./resnet50_ham10000_model2.pth", map_location=device))
model.to(device)
model.eval()

# Define transformations (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Label mapping (updated for your CSV classes)
label_map = {
    "bkl": "Benign Keratosis",
    "nv": "Melanocytic Nevus",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "vasc": "Vascular Lesion",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis"
}

# Load CSV with actual labels
CSV_PATH = "/Users/avirajsingh/Desktop/IITB/Workshop/Interface/Sample Images/Labels/sample_images.csv"  # Ensure this matches your file location
df_labels = pd.read_csv(CSV_PATH)

def predict_image(file):
    """Process image, predict label, and retrieve actual label from CSV."""
    # Save uploaded image
    file_ext = file.name.split('.')[-1]
    filename = f"{uuid.uuid4()}.{file_ext}"  # Generate unique filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file.getvalue())  # Save the uploaded file

    try:
        # Load and preprocess image
        image = Image.open(file_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Run model prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_idx = predicted_idx.item()

        # Map predicted index to class name (assuming model outputs 0-6)
        index_to_label = {i: label for i, label in enumerate(label_map.keys())}
        predicted_class_short = index_to_label.get(predicted_idx, "Unknown")
        predicted_class = label_map.get(predicted_class_short, "Unknown")

        # Extract image ID (e.g., "ISIC_0027419" from "ISIC_0027419.jpg")
        image_id = file.name.split('.')[0]  # Remove file extension

        # Find actual label from CSV
        actual_class_short = df_labels[df_labels["image_id"] == image_id]["dx"]
        actual_class_short = actual_class_short.iloc[0] if not actual_class_short.empty else "N/A (Not found in CSV)"
        actual_class = label_map.get(actual_class_short, actual_class_short)

        return file_path, predicted_class, actual_class

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return file_path, None, None

# Streamlit UI
st.title("Skin Lesion Classifier")
st.write("Upload an image to see the predicted and actual labels based on the HAM10000 dataset.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Process and predict
    with st.spinner("Classifying..."):
        file_path, predicted_label, actual_label = predict_image(uploaded_file)
        
        if predicted_label and actual_label:
            st.subheader("Results")
            st.write(f"**Predicted Label**: {predicted_label}")
            st.write(f"**Actual Label**: {actual_label}")
            if predicted_label == actual_label:
                st.success("The prediction matches the actual label!")
            else:
                st.warning("The prediction does not match the actual label.")
            
            # Display the saved image path (optional)
            st.write(f"Image saved at: `{file_path}`")
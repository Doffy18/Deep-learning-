import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import DistilBertTokenizerFast, DistilBertModel
import os

from multimodal_model import MultimodalClassifier

# --------------------- CONFIG ---------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)

# Image transform (must match precomputing_embedding.py)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --------------------- LOAD MODELS ---------------------
@st.cache_resource
def load_encoders():
    # Load ResNet50 backbone for image embeddings
    resnet = models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(DEVICE).eval()

    # Load DistilBERT for text embeddings
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    distilbert = distilbert.to(DEVICE).eval()

    return resnet, tokenizer, distilbert

@st.cache_resource
def load_classifier(model_path, labelmap_path):
    data = torch.load(labelmap_path, map_location=DEVICE)
    label2id = data['label2id']
    id2label = {v: k for k, v in label2id.items()}

    model = MultimodalClassifier(num_labels=len(label2id))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model, id2label

# --------------------- INFERENCE UTILS ---------------------
def embed_image(image, resnet):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_feat = resnet(img_tensor)
    return img_feat

def embed_text(text, tokenizer, distilbert):
    encoded = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)
    with torch.no_grad():
        outputs = distilbert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs.last_hidden_state[:, 0, :]  # CLS token
    return text_feat

def predict(image_embedding, text_embedding, model, id2label):
    with torch.no_grad():
        outputs = model(image_embedding, text_embedding)
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        return id2label[pred_idx], float(probs[0, pred_idx])

# --------------------- STREAMLIT APP ---------------------
def main():
    st.set_page_config(page_title="Multimodal Classifier", page_icon="🖼️")
    st.title("🖼️ Multimodal Product Classifier")
    st.markdown("""
    ## Introduction
    This app classifies products using both **images** and **text descriptions**.
    Images are processed by a ResNet50 model, text is processed by DistilBERT,
    and their embeddings are fused by a trained classifier.
    """)

    # Load models
    resnet, tokenizer, distilbert = load_encoders()
    model, id2label = load_classifier("model.pt", "test_embeddings.pt")

    # User inputs
    uploaded_image = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
    text_description = st.text_area("Enter product description")

    if st.button("Predict"):
        if not uploaded_image or not text_description.strip():
            st.error("Please provide both an image and a description.")
            return
        
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Compute embeddings
        img_embed = embed_image(image, resnet)
        txt_embed = embed_text(text_description, tokenizer, distilbert)

        # Predict
        label, confidence = predict(img_embed, txt_embed, model, id2label)

        st.success(f"**Predicted Category:** {label} \n\n **Confidence:** {confidence:.2%}")

if __name__ == "__main__":
    main()

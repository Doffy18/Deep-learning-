import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from transformers import DistilBertTokenizerFast, DistilBertModel
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

CSV_PATHS = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv"
}
IMG_FOLDER = "images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image model (ResNet50 backbone)
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()

# Text model (DistilBERT)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
distilbert = distilbert.to(device).eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def embed_split(split_name, csv_path):
    df = pd.read_csv(csv_path)
    image_embeds = []
    text_embeds = []
    labels = []

    batch_size = 64  
    num_samples = len(df)

    print(f"Processing {split_name} split embeddings for {num_samples} samples...")

    for start_idx in tqdm(range(0, num_samples, batch_size)):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_df = df.iloc[start_idx:end_idx]

        # Process images
        imgs = []
        for img_name in batch_df['image']:
            img_path = os.path.join(IMG_FOLDER, img_name)
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            imgs.append(image)
        imgs_tensor = torch.stack(imgs).to(device)

        with torch.no_grad():
            img_feats = resnet(imgs_tensor)  # shape: [batch_size, 2048]

        # Process text
        texts = batch_df['description'].fillna("").tolist()
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            text_outputs = distilbert(input_ids=input_ids, attention_mask=attention_mask)
            text_feats = text_outputs.last_hidden_state[:, 0, :]  # CLS token, shape: [batch_size, 768]

        image_embeds.append(img_feats.cpu())
        text_embeds.append(text_feats.cpu())

        # Process labels
        labels.extend(batch_df['category'].tolist())

    # Concatenate all batches
    image_embeds = torch.cat(image_embeds)
    text_embeds = torch.cat(text_embeds)

    # Create label mapping identical to training
    unique_labels = sorted(list(set(labels)))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    label_ids = torch.tensor([label2id[l] for l in labels])

    # Save embeddings and labels
    save_path = f"{split_name}_embeddings.pt"
    torch.save({
        "image_embeds": image_embeds,
        "text_embeds": text_embeds,
        "labels": label_ids,
        "label2id": label2id
    }, save_path)
    print(f"Saved {split_name} embeddings to {save_path}")
    return label2id

if __name__ == "__main__":
    # Run for all splits
    label2id_global = None
    for split in ["train", "val", "test"]:
        label2id = embed_split(split, CSV_PATHS[split])
        if label2id_global is None:
            label2id_global = label2id
        else:
            assert label2id == label2id_global, "Label maps differ between splits!"

import torch
from torch.utils.data import Dataset

class CachedMultimodalDataset(Dataset):
    def __init__(self, embedding_file):
        data = torch.load(embedding_file)
        self.image_embeds = data['image_embeds']  # Tensor [N, 2048]
        self.text_embeds = data['text_embeds']    # Tensor [N, 768]
        self.labels = data['labels']              # Tensor [N]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "pixel_values": self.image_embeds[idx],   # Keep name for Trainer compatibility
            "input_ids": self.text_embeds[idx],
            "attention_mask": torch.ones(self.text_embeds[idx].shape[0]),  # Dummy mask
            "labels": self.labels[idx]
        }




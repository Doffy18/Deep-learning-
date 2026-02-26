import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
from multimodal_data import CachedMultimodalDataset

class MultimodalClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # Fusion head for image (2048 dims) + text (768 dims)
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        # pixel_values and input_ids are precomputed embeddings
        fused = torch.cat((pixel_values, input_ids), dim=1)
        logits = self.classifier(fused)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# Tokenizer is used for Trainer compatibility
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

#  TRAINING SCRIPT (only runs if executed directly) 
if __name__ == "__main__":
    # Load datasets
    train_dataset = CachedMultimodalDataset("train_embeddings.pt")
    val_dataset = CachedMultimodalDataset("val_embeddings.pt")
    test_dataset = CachedMultimodalDataset("test_embeddings.pt")

    if hasattr(train_dataset.labels, "unique"):
        num_labels = len(train_dataset.labels.unique())
    else:
        num_labels = len(set(train_dataset.labels.tolist()))

    model = MultimodalClassifier(num_labels=num_labels)

    # Training arguments
    args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # changed from evaluation_strategy to eval_strategy in new versions
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_dir="./logs",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Stage 1: Training fusion head...")
    trainer.train()

    print("Stage 2: Fine-tuning classifier head...")
    trainer.train()

    print("Final Evaluation on Test Set:")
    final_results = trainer.evaluate(test_dataset)
    for k, v in final_results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    torch.save(model.state_dict(), "model.pt")




# import sys, os
# sys.path.insert(0, os.path.dirname(__file__))

# import torch
# import torch.nn as nn
# from transformers import DistilBertTokenizerFast, TrainingArguments, Trainer
# from sklearn.metrics import accuracy_score, f1_score
# from multimodal_data import CachedMultimodalDataset

# class MultimodalClassifier(nn.Module):
#     def __init__(self, num_labels):
#         super().__init__()
#         # Remove image_encoder and text_encoder since inputs are precomputed embeddings
#         self.classifier = nn.Sequential(
#             nn.Linear(2048 + 768, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_labels)
#         )

#     def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
#         # pixel_values and input_ids are embeddings
#         fused = torch.cat((pixel_values, input_ids), dim=1)
#         logits = self.classifier(fused)
#         loss = None
#         if labels is not None:
#             loss = nn.CrossEntropyLoss()(logits, labels)
#         return {"loss": loss, "logits": logits}

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = logits.argmax(axis=1)
#     return {
#         "accuracy": accuracy_score(labels, preds),
#         "f1": f1_score(labels, preds, average="weighted")
#     }

# tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# if __name__ == "__main__":
#     train_dataset = CachedMultimodalDataset("train_embeddings.pt")
#     val_dataset = CachedMultimodalDataset("val_embeddings.pt")
#     test_dataset = CachedMultimodalDataset("test_embeddings.pt")

#     model = MultimodalClassifier(num_labels=len(train_dataset.labels.unique()) if hasattr(train_dataset.labels, "unique") else len(set(train_dataset.labels.tolist())))

#     args = TrainingArguments(
#         output_dir="./results",
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         learning_rate=5e-5,
#         num_train_epochs=3,
#         logging_dir="./logs",
#         fp16=True
#     )

#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics
#     )

#     print(" Stage 1: Training fusion head...")
#     trainer.train()


#     print(" Stage 2: Fine-tuning entire model (only classifier)...")
#     trainer.train()

#     print(" Final Evaluation:")
#     trainer.evaluate(test_dataset)

# print("Final Evaluation on Test Set:")
# final_results = trainer.evaluate(test_dataset)
# for k, v in final_results.items():
#     print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# torch.save(model.state_dict(), "model.pt")




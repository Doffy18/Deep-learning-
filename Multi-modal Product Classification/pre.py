import pandas as pd
from sklearn.model_selection import train_test_split
import os

CSV_PATH = "data.csv"
IMG_FOLDER = "images"  

df = pd.read_csv(CSV_PATH)
initial_count = len(df)
df = df.dropna()

# remove rare categories (<4 samples) 
rare_df = df.groupby('category').filter(lambda x: len(x) < 8)
df = df.groupby('category').filter(lambda x: len(x) >= 8)
final_count = len(df)
removed_count = initial_count - final_count

#  Downsample dataset to 30k samples total, stratified by category 
total_target = 30000
if len(df) > total_target:
    fraction = total_target / len(df)
    df = df.groupby('category', group_keys=False).apply(lambda x: x.sample(frac=fraction, random_state=42))

#  Split into train (20k) and temp (10k) sets, stratified 
train_df, temp_df = train_test_split(
    df,
    test_size=10000,
    stratify=df['category'],
    random_state=42
)

# Split temp into validation and test sets (5k each), stratified
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['category'],
    random_state=42
)

# Save removed info 
rare_df.to_csv("removed_samples.csv", index=False)
rare_df['image'].to_csv("images_to_delete.txt", index=False, header=False)

# Delete removed images 
removed_images = rare_df['image'].tolist()
deleted_count = 0
for img_name in removed_images:
    img_path = os.path.join(IMG_FOLDER, img_name)
    if os.path.exists(img_path):
        os.remove(img_path)
        deleted_count += 1
print(f"🗑 Deleted {deleted_count} images from '{IMG_FOLDER}'.")

# Save final splits
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Initial size: {initial_count}")
print(f"Final size after removing rare categories: {final_count}")
print(f"Removed samples: {removed_count}")
print(f"Rare categories removed: {rare_df['category'].nunique()}")
print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

print(" Train/Val/Test CSVs saved successfully.")
print(" Removed samples saved in 'removed_samples.csv'.")
print(" Deleted images list saved in 'images_to_delete.txt'.")






import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import torch

# Load the dataset
input_file = r"C:\projects\processed_dataset.csv"  # Replace with your file path
df = pd.read_csv(input_file)

# Columns in the dataset
caption_column = "caption"  # Adjust to match the column name in your file

# Load a pre-trained SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")  # You can use other models if needed

# Generate embeddings for captions
df["caption_embedding"] = df[caption_column].apply(lambda x: model.encode(x).tolist())

# Save the processed dataset
output_file = r"C:\projects\processed_with_embeddings.csv"
df.to_csv(output_file, index=False)

print(f"Processed dataset with embeddings saved to {output_file}.")

# Column containing mel spectrogram paths
mel_spectrogram_path_column = "mel_spectrogram_path"

# Parameters for preprocessing
target_size = (128, 256)  # Target shape for downsampling (freq_bins, time_steps)

def normalize_spectrogram(spectrogram):
    """
    Normalize a spectrogram to have zero mean and unit variance.
    """
    mean = torch.mean(spectrogram)
    std = torch.std(spectrogram)
    return (spectrogram - mean) / (std + 1e-6)  # Add epsilon to avoid division by zero

def downsample_spectrogram(spectrogram, target_shape):
    """
    Downsample a spectrogram to the target shape using interpolation.
    """
    return torch.nn.functional.interpolate(
        spectrogram.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
        size=target_shape,
        mode="bilinear",
        align_corners=False
    ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions

def preprocess_mel_spectrogram(path, target_shape):
    """
    Load, normalize, and downsample a mel spectrogram.
    """
    # Load spectrogram
    spectrogram = torch.load(path)  # Assumes .pt format
    if spectrogram.ndim == 2:
        spectrogram = spectrogram.unsqueeze(0)  # Add channel dimension if missing

    # Normalize
    spectrogram = normalize_spectrogram(spectrogram)

    # Downsample
    spectrogram = downsample_spectrogram(spectrogram, target_shape)

    return spectrogram

# Preprocess all mel spectrograms
preprocessed_data = []
output_dir = "preprocessed_mel_spectrograms"  # Directory to save preprocessed spectrograms
os.makedirs(output_dir, exist_ok=True)

for index, row in df.iterrows():
    mel_path = row[mel_spectrogram_path_column]
    try:
        # Preprocess the mel spectrogram
        processed_spectrogram = preprocess_mel_spectrogram(mel_path, target_size)

        # Save the processed spectrogram
        output_path = os.path.join(output_dir, f"preprocessed_{os.path.basename(mel_path)}")
        torch.save(processed_spectrogram, output_path)

        # Append processed path to the DataFrame
        preprocessed_data.append(output_path)
    except Exception as e:
        print(f"Error processing {mel_path}: {e}")
        preprocessed_data.append(None)

# Add preprocessed paths to the DataFrame
df["preprocessed_mel_spectrogram_path"] = preprocessed_data

# Save the updated dataset
output_file = "processed_with_audio.csv"
df.to_csv(output_file, index=False)

print(f"Updated dataset saved to {output_file}.")

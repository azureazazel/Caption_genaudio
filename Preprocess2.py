import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import torch

# Load the dataset
input_file = r"C:\projects\processed_dataset.csv"  
df = pd.read_csv(input_file)


caption_column = "caption"  


model = SentenceTransformer("all-MiniLM-L6-v2")  


df["caption_embedding"] = df[caption_column].apply(lambda x: model.encode(x).tolist())


output_file = r"C:\projects\processed_with_embeddings.csv"
df.to_csv(output_file, index=False)

print(f"Processed dataset with embeddings saved to {output_file}.")


mel_spectrogram_path_column = "mel_spectrogram_path"


target_size = (128, 256)  

def normalize_spectrogram(spectrogram):
    """
    Normalize a spectrogram to have zero mean and unit variance.
    """
    mean = torch.mean(spectrogram)
    std = torch.std(spectrogram)
    return (spectrogram - mean) / (std + 1e-6)  

def downsample_spectrogram(spectrogram, target_shape):
    """
    Downsample a spectrogram to the target shape using interpolation.
    """
    return torch.nn.functional.interpolate(
        spectrogram.unsqueeze(0).unsqueeze(0),  
        size=target_shape,
        mode="bilinear",
        align_corners=False
    ).squeeze(0).squeeze(0)  

def preprocess_mel_spectrogram(path, target_shape):
    """
    Load, normalize, and downsample a mel spectrogram.
    """
    
    spectrogram = torch.load(path) 
    if spectrogram.ndim == 2:
        spectrogram = spectrogram.unsqueeze(0)  

    
    spectrogram = normalize_spectrogram(spectrogram)

   
    spectrogram = downsample_spectrogram(spectrogram, target_shape)

    return spectrogram


preprocessed_data = []
output_dir = "preprocessed_mel_spectrograms"  
os.makedirs(output_dir, exist_ok=True)

for index, row in df.iterrows():
    mel_path = row[mel_spectrogram_path_column]
    try:
      
        processed_spectrogram = preprocess_mel_spectrogram(mel_path, target_size)

        
        output_path = os.path.join(output_dir, f"preprocessed_{os.path.basename(mel_path)}")
        torch.save(processed_spectrogram, output_path)

        
        preprocessed_data.append(output_path)
    except Exception as e:
        print(f"Error processing {mel_path}: {e}")
        preprocessed_data.append(None)


df["preprocessed_mel_spectrogram_path"] = preprocessed_data


output_file = "processed_with_audio.csv"
df.to_csv(output_file, index=False)

print(f"Updated dataset saved to {output_file}.")

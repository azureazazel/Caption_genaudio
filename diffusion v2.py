import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_spectrogram(data_path, target_length=430):
    spectrogram = torch.load(data_path).numpy()
    
    if spectrogram.shape[1] > target_length:
        spectrogram = spectrogram[:, :target_length]
    else:
        spectrogram = np.pad(spectrogram, ((0, 0), (0, target_length - spectrogram.shape[1])), mode='constant')
    return spectrogram


class AudioDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.caption_embeddings = dataframe['caption_embedding'].apply(json.loads).values
        self.mel_spectrogram_paths = dataframe['preprocessed_mel_spectrogram_path'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        caption_embedding = torch.tensor(self.caption_embeddings[idx], dtype=torch.float32)
        spectrogram_path = self.mel_spectrogram_paths[idx]
        spectrogram = torch.load(spectrogram_path)  # Load spectrogram

        # Ensure spectrogram has 1 channel
        return caption_embedding, spectrogram.unsqueeze(0) 


import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1,embedding_dim=512):
        super(UNet, self).__init__()
        # Define your encoder and decoder layers here.
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.embedding_fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, spectrogram, caption_embedding):
        encoded_spectrogram = self.encoder(spectrogram)
        
        # Process the caption embedding
        caption_features = self.embedding_fc(caption_embedding)
        caption_features = caption_features.view(caption_features.size(0), 128, 1, 1)  # Reshape to [B, 128, 1, 1]

        # Add the caption features to the spectrogram encoding
        conditioned_features = encoded_spectrogram + caption_features
        
        # Decode the conditioned features
        return self.decoder(conditioned_features)
def add_noise(spectrogram, t, beta_schedule):
    """
    Adds noise to the spectrogram for a given timestep t.
    """
    beta_t = beta_schedule[t]
    noise = torch.randn_like(spectrogram)  # Same shape as spectrogram
    noisy_spectrogram = (1 - beta_t).sqrt() * spectrogram + beta_t.sqrt() * noise
    return noisy_spectrogram, noise

def get_beta_schedule(timesteps, start=1e-4, end=2e-2):
    """
    Creates a linear beta schedule.

    Args:
    - timesteps (int): Number of timesteps.
    - start (float): Starting beta value.
    - end (float): Ending beta value.

    Returns:
    - Beta schedule (torch.Tensor).
    """
    return torch.linspace(start, end, timesteps)

loss_fn = nn.MSELoss()

def compute_loss(model, spectrogram, t, beta_schedule):
    """
    Computes the loss for a single training step.

    Args:
    - model (nn.Module): UNet denoising model.
    - spectrogram (torch.Tensor): Clean spectrogram [B, C, H, W].
    - t (int): Current timestep.
    - beta_schedule (torch.Tensor): Noise schedule.

    Returns:
    - Loss (torch.Tensor).
    """
    noisy_spectrogram, noise = add_noise(spectrogram, t, beta_schedule)
    predicted_noise = model(noisy_spectrogram)
    loss = loss_fn(predicted_noise, noise)
    return loss

def train_diffusion_model(model, dataloader, timesteps, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    beta_schedule = get_beta_schedule(timesteps).to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for caption_embeddings, spectrograms in dataloader:
            caption_embeddings = caption_embeddings.to(device)  # [batch_size, embedding_dim]
            spectrograms = spectrograms.to(device)  # [batch_size, 1, 128, 430]

            t = torch.randint(0, timesteps, (1,)).item()  # Random timestep
            noisy_spectrogram, noise = add_noise(spectrograms, t, beta_schedule)  # [batch_size, 1, 128, 430]

            optimizer.zero_grad()
            predicted_noise = model(noisy_spectrogram, caption_embeddings)  # Pass spectrogram and caption embeddings
            loss = loss_fn(predicted_noise, noise)  # Compute loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

def denoise(model, noisy_spectrogram, caption_embedding, timesteps, beta_schedule):
    """
    Iteratively denoise a spectrogram.
    
    Args:
    - model (nn.Module): Trained UNet model.
    - noisy_spectrogram (torch.Tensor): Initial noisy spectrogram.
    - timesteps (int): Number of timesteps.
    - beta_schedule (torch.Tensor): Noise schedule.

    Returns:
    - Cleaned spectrogram (torch.Tensor).
    """
    for t in reversed(range(timesteps)):
        beta_t = beta_schedule[t]
        predicted_noise = model(noisy_spectrogram, caption_embedding)
        noisy_spectrogram = (noisy_spectrogram - beta_t.sqrt() * predicted_noise) / (1 - beta_t).sqrt()
    return noisy_spectrogram


 #Griffin-Lim for Spectrogram-to-Audio
import librosa
import pandas as pd
import soundfile as sf
def spectrogram_to_audio_griffinlim(spectrogram, sr=22050, n_iter=32, n_fft=1024, hop_length=256):
    spectrogram = spectrogram.squeeze(0).cpu().numpy()  # Remove batch dimension
    spectrogram = np.clip(spectrogram, a_min=1e-10, a_max=None)  # Avoid log(0)
    waveform = librosa.griffinlim(spectrogram, n_iter=n_iter, hop_length=hop_length, win_length=n_fft)
    return waveform

# Load Data
data_path = r"C:\Users\rahul\OneDrive\Documents\DS_ML projects\merged_datasetv3.csv"
df = pd.read_csv(data_path)

# Dataset and DataLoader
dataset = AudioDataset(df)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model Initialization
timesteps = 500
embedding_dim = len(json.loads(df['caption_embedding'].iloc[0]))  # Size of caption embeddings
model = UNet(input_channels=1, output_channels=1, embedding_dim=embedding_dim).to(device)

# Train the Model
train_diffusion_model(model, dataloader, timesteps=timesteps, epochs=5, lr=1e-4)

# Inference
beta_schedule = get_beta_schedule(timesteps).to(device)
noisy_input = torch.randn(1, 1, 128, 430).to(device)  # Random noise as input
caption_embedding = torch.tensor(json.loads(df['caption_embedding'].iloc[5]), dtype=torch.float32).unsqueeze(0).to(device)
generated_spectrogram = denoise(model, noisy_input, caption_embedding, timesteps, beta_schedule)

# Save Generated Spectrogram
save_path = r"C:\Users\rahul\OneDrive\Documents\DS_ML projects\generated spectrograms\generated_spectrogram_with_caption.pt"
torch.save(generated_spectrogram, save_path)
print(f"Generated spectrogram saved to {save_path}")

# Convert to Audio
spectrogram = generated_spectrogram.squeeze(0)  # Remove batch dimension
audio_waveform = spectrogram_to_audio_griffinlim(spectrogram, sr=22050)
output_path = "generated_audio_with_caption.wav"
sf.write(output_path, audio_waveform, samplerate=22050)
print(f"Audio saved to {output_path}")
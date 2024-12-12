import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_spectrogram(file_path, target_length=256):
    spectrogram = torch.load(file_path).numpy()
    
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
        embedding = torch.tensor(self.caption_embeddings[idx], dtype=torch.float32)
        spectrogram_path = self.mel_spectrogram_paths[idx]
        spectrogram = torch.tensor(load_spectrogram(spectrogram_path), dtype=torch.float32)
        return embedding, spectrogram


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(x).reshape(-1, 128, 256)  


def train_model(encoder, decoder, dataloader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    encoder.train()
    decoder.train()

    for epoch in range(epochs):
        total_loss = 0
        for embeddings, spectrograms in dataloader:
            embeddings = embeddings.to(device)
            spectrograms = spectrograms.to(device)

            optimizer.zero_grad()
            latent = encoder(embeddings)
            reconstructed_spectrograms = decoder(latent)

            loss = loss_fn(reconstructed_spectrograms, spectrograms)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


import pandas as pd

data_path = r"C:\Users\rahul\OneDrive\Documents\DS_ML projects\merged_dataset1.csv" # the spreadsheet provided in zip file 
df = pd.read_csv(data_path)

embedding_dim = len(json.loads(df['caption_embedding'].iloc[0]))  
latent_dim = 128
output_dim = 128 * 256  

encoder = Encoder(embedding_dim, latent_dim).to(device)
decoder = Decoder(latent_dim, output_dim).to(device)



# Save the Model
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(decoder.state_dict(), "decoder.pth")

def generate_audio(caption_embedding, save_path=r"C:\Users\rahul\OneDrive\Documents\DS_ML projects\generated spectrograms\generated_spectrogram1.pt"):# Your local drive path
    """
    Generate a spectrogram from a caption embedding and save it as a .pt file.
    """
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        caption_embedding = torch.tensor(caption_embedding, dtype=torch.float32).to(device)
        latent = encoder(caption_embedding)
        generated_spectrogram = decoder(latent)

    # Save the spectrogram
    torch.save(generated_spectrogram, save_path)
    print(f"Generated spectrogram saved to {save_path}")
    return generated_spectrogram


import torch
import numpy as np
import librosa
import soundfile as sf

# Griffin-Lim for converting spectrogram to audio
def spectrogram_to_audio_griffinlim(spectrogram, sr=22050, n_iter=32, n_fft=254, hop_length=256):
    """
    Converts a spectrogram to audio using the Griffin-Lim algorithm.
    """
    spectrogram = spectrogram.squeeze(0).cpu().numpy()  # Remove batch dimension if necessary
    spectrogram = np.clip(spectrogram, a_min=1e-10, a_max=None)  # Avoid log(0)
    waveform = librosa.griffinlim(spectrogram, n_iter=n_iter, hop_length=hop_length, win_length=n_fft)
    return waveform


# Preprocessing
dataset = AudioDataset(df)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

caption_embedding = json.loads(df['caption_embedding'].iloc[5]) # you can change the caption embedding to use 

train_model(encoder, decoder, dataloader, epochs=20, lr=1e-4)
generated_spectrogram = generate_audio(caption_embedding, save_path=r"C:\Users\rahul\OneDrive\Documents\DS_ML projects\generated spectrograms\generated_spectrogram1.pt")




if __name__ == "__main__":
    # Load the saved spectrogram
    generated_spectrogram = torch.load(r"C:\Users\rahul\OneDrive\Documents\DS_ML projects\generated spectrograms\generated_spectrogram1.pt"
                                       )
    print(f"Loaded spectrogram shape: {generated_spectrogram.shape}")  # [1, 128, T]

    
    spectrogram = generated_spectrogram.squeeze(0)  # Shape becomes [128, T]

    
    print("Converting spectrogram to audio...")
    audio_waveform = spectrogram_to_audio_griffinlim(spectrogram, sr=22050)

   
    output_path = "generated_audio_griffinlim.wav"
    sf.write(output_path, audio_waveform, samplerate=22050)
    print(f"Audio saved to {output_path}")


import os
import pandas as pd
import torch

# Path to the captions file (update with your file's path)
captions_file_path = r"C:\Users\rahul\OneDrive\Documents\DS_ML projects\Final\archive\musiccaps-public.csv"  
mel_spectrogram_dir = r"C:\Users\rahul\OneDrive\Documents\DS_ML projects\Final\output_tensors1"  
output_file = r"C:\projects\combined_data.csv"


df = pd.read_csv(captions_file_path)  # Use pd.read_excel() if Excel file

ytid_column = "ytid"
caption_column = "caption"  

if ytid_column not in df.columns or caption_column not in df.columns:
    raise ValueError(f"Ensure '{ytid_column}' and '{caption_column}' are columns in your file.")


pt_files = [f for f in os.listdir(mel_spectrogram_dir) if f.endswith(".pt")]


matched_data = []
for _, row in df.iterrows():
    ytid = row[ytid_column]
    caption = row[caption_column]
    # Generate the expected .pt filename
    pt_filename = f"{ytid}.pt"
    pt_path = os.path.join(mel_spectrogram_dir, pt_filename)

    # Check if the corresponding .pt file exists
    if pt_filename in pt_files:
        matched_data.append({"ytid": ytid, "caption": caption, "mel_spectrogram_path": pt_path})


processed_df = pd.DataFrame(matched_data)


processed_df.to_csv(output_file, index=False)

print(f"Processed dataset saved to {output_file}.")

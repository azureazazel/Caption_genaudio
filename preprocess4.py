import pandas as pd


embeddings_file = r"C:\projects\processed_with_embeddings.csv"  
mel_spectrograms_file = r"C:\Users\rahul\OneDrive\Documents\DS_ML projects\final processed.csv"  


embeddings_df = pd.read_csv(embeddings_file)
mel_spectrograms_df = pd.read_csv(mel_spectrograms_file)


common_column = "ytid"
if common_column not in embeddings_df.columns or common_column not in mel_spectrograms_df.columns:
    raise ValueError(f"Ensure both files have a common column '{common_column}' for merging.")


merged_df = pd.merge(embeddings_df, mel_spectrograms_df, on=common_column, how="inner")


output_file = "merged_datasetv3.csv"
merged_df.to_csv(output_file, index=False)

print(f"Merged dataset saved to {output_file}.")

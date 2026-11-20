### طالب احمد راجي سعيد عميره  2400902065
import os
import re
import pandas as pd
from datasets import load_dataset

def generate_smart_title(text: str, stars: int) -> str:
    clean_text = text.replace("\\n", " ").replace('\n', ' ').replace('\\"', '"').strip()
    first_sentence = re.split(r'[.!?]+', clean_text)[0].strip()
    words = first_sentence.split()

    if len(words) >= 3:
        snippet = " ".join(words[:7])
        if len(words) > 7:
            snippet += "..."
    else:
        snippet = " ".join(clean_text.split()[:7]) + "..."

    return f"{snippet} ({stars} Stars)"

def build_dataset(output_path="IR_Project/dataset/documents.csv", sample_size=1000, random_state=42):
    print("\n" + "═"*80)
    print("Downloading dataset 'Yelp/yelp_review_full' from HuggingFace...")
    print("═"*80 + "\n")
    
    dataset = load_dataset("Yelp/yelp_review_full", split="test")
    df = pd.DataFrame(dataset)

    # Clean text column
    df['text'] = df['text'].astype(str).str.replace(r'\\n|\n|\r', ' ', regex=True).str.strip()
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)

    # Remove missing/empty records and very short meaningless strings
    df = df[df['text'].str.len() > 25]

    # Remove exact duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=['text']).copy()
    print(f"\nRemoved {initial_len - len(df)} duplicate records.\n")

    # Convert label (0-4) to actual Star Rating (1-5)
    df['stars'] = df['label'] + 1

    # Perform Stratified Reproducible Sampling
    samples_per_class = sample_size // 5
    sampled_df = (
        df.groupby('stars', group_keys=False)
        .sample(n=samples_per_class, random_state=random_state)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    sampled_df['doc_id'] = [f"D{i+1}" for i in range(len(sampled_df))]
    sampled_df['category'] = sampled_df['stars'].apply(lambda s: f"{s}-Star Review")
    sampled_df['title'] = [generate_smart_title(row['text'], row['stars']) for _, row in sampled_df.iterrows()]

    final_df = sampled_df[['doc_id', 'title', 'category', 'text']]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved {len(final_df)} documents successfully to: '{output_path}'\n")
    return final_df

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "dataset", "documents.csv")
    df_out = build_dataset(output_path=csv_path, sample_size=1000, random_state=42)
    
    print("┌" + "─"*78 + "┐")
    print("│ Dataset Sample (First 5 Documents)                                           │")
    print("├" + "─"*78 + "┤")
    print(df_out[['doc_id', 'title', 'category']].head(5).to_string(index=False))
    print("└" + "─"*78 + "┘\n")

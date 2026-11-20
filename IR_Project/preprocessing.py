import os
import re
from typing import List, Dict, Any
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class TextPreprocessor:
    def __init__(self, remove_stopwords: bool = True, normalization: str = "lemmatize", custom_stopwords: List[str] = None):
        if normalization not in {"lemmatize", "stem", "none"}:
            raise ValueError("normalization must be 'lemmatize', 'stem', or 'none'")

        self.remove_stopwords = remove_stopwords
        self.normalization = normalization

        self.stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            self.stop_words.update(w.lower() for w in custom_stopwords)

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def tokenize_only(self, text: str) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        return word_tokenize(cleaned)

    def preprocess(self, text: str) -> List[str]:
        tokens = self.tokenize_only(text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        else:
            tokens = [t for t in tokens if len(t) > 1]

        if self.normalization == "lemmatize":
            return [self.lemmatizer.lemmatize(t) for t in tokens]
        elif self.normalization == "stem":
            return [self.stemmer.stem(t) for t in tokens]
        return tokens

    def inspect_pipeline(self, text: str) -> Dict[str, Any]:
        raw_tokens = self.tokenize_only(text)
        filtered_tokens = [t for t in raw_tokens if t not in self.stop_words and len(t) > 1]
        lemmatized = [self.lemmatizer.lemmatize(t) for t in filtered_tokens]
        stemmed = [self.stemmer.stem(t) for t in filtered_tokens]

        return {
            "original_text": text,
            "raw_tokens": raw_tokens,
            "filtered_tokens_no_stopwords": filtered_tokens,
            "lemmatized_terms": lemmatized,
            "stemmed_terms": stemmed
        }

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "dataset", "documents.csv")
    df = pd.read_csv(csv_path)
    sample_doc = df.iloc[0]

    prep = TextPreprocessor(remove_stopwords=True, normalization="lemmatize")
    inspection = prep.inspect_pipeline(sample_doc['text'])

    print("\n" + "═"*80)
    print("PREPROCESSING MODULE")
    print("═"*80 + "\n")
    
    print(f"Document ID : {sample_doc['doc_id']}")
    print(f"Original Text :\n   {inspection['original_text'][:200]}...\n")
    
    print("──────────── Pipeline Steps ────────────\n")
    print(f"1️⃣ Raw Tokens (Total: {len(inspection['raw_tokens'])})  :\n   {inspection['raw_tokens'][:10]} ...\n")
    print(f"2️⃣ No Stopwords (Total: {len(inspection['filtered_tokens_no_stopwords'])})  :\n   {inspection['filtered_tokens_no_stopwords'][:10]} ...\n")
    print(f"3️⃣ Lemmatized (Total: {len(inspection['lemmatized_terms'])})  :\n   {inspection['lemmatized_terms'][:10]} ...\n")
    print(f"4️⃣ Stemmed (Total: {len(inspection['stemmed_terms'])})  :\n   {inspection['stemmed_terms'][:10]} ...\n")
    print("═"*80 + "\n")

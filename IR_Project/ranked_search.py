import os
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import TextPreprocessor

class RankedSearchEngine:
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.vectorizer = TfidfVectorizer(tokenizer=self.preprocessor.preprocess, token_pattern=None, lowercase=False)
        self.tfidf_matrix, self.df, self.feature_names = None, None, None

    def fit_corpus(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['text'])
        self.feature_names = self.vectorizer.get_feature_names_out()

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            return {"query": query, "status": "empty_query", "results": []}

        processed_query_terms = self.preprocessor.preprocess(query)
        if not processed_query_terms:
            return {"query": query, "status": "no_valid_terms", "results": []}

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for rank_idx, doc_idx in enumerate(sorted_indices[:top_k], start=1):
            score = float(similarities[doc_idx])
            if score <= 0.0: break
            row = self.df.iloc[doc_idx]
            results.append({
                "rank": rank_idx,
                "doc_id": str(row['doc_id']),
                "title": str(row['title']),
                "score": round(score, 4),
            })

        return {"query": query, "processed_query": processed_query_terms, "status": "success", "results": results}

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "dataset", "documents.csv")
    df = pd.read_csv(csv_path)
    prep = TextPreprocessor(remove_stopwords=True, normalization="lemmatize")
    rank_eng = RankedSearchEngine(preprocessor=prep)
    rank_eng.fit_corpus(df)

    print("\n" + "═"*80)
    print("RANKED TF-IDF SEARCH")
    print("═"*80 + "\n")
    print(f"TF-IDF Matrix Shape: {rank_eng.tfidf_matrix.shape}\n")

    for q in ["meatballs Chinese tasty food", "spicy burger", "fresh sushi rolls"]:
        res = rank_eng.search(q, top_k=3)
        print(f"Query: '{q}'")
        print(f"Processed: {res['processed_query']}")
        print("  ┌──────┬──────────┬──────────────┬─────────────────────────────────────────────┐")
        print(f"  │ {'Rank':<4} │ {'Doc ID':<8} │ {'Cosine Score':<12} │ {'Title':<43} │")
        print("  ├──────┼──────────┼──────────────┼─────────────────────────────────────────────┤")
        for item in res['results']:
            title_display = item['title'][:40] + "..." if len(item['title']) > 40 else item['title']
            print(f"  │ #{item['rank']:<3} │ {item['doc_id']:<8} │ {item['score']:<12.4f} │ {title_display:<43} │")
        print("  └──────┴──────────┴──────────────┴─────────────────────────────────────────────┘\n")

import os
from typing import Dict, Set, List
import pandas as pd
from preprocessing import TextPreprocessor

class InvertedIndex:
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.index: Dict[str, Set[str]] = {}
        self.all_doc_ids: Set[str] = set()
        self.doc_metadata: Dict[str, Dict[str, str]] = {}

    def build_index(self, df: pd.DataFrame) -> None:
        self.index.clear()
        self.all_doc_ids.clear()
        self.doc_metadata.clear()

        for _, row in df.iterrows():
            doc_id = str(row['doc_id'])
            self.all_doc_ids.add(doc_id)
            self.doc_metadata[doc_id] = {
                "title": str(row['title']),
                "category": str(row['category']),
                "text": str(row['text'])
            }
            terms = self.preprocessor.preprocess(str(row['text']))
            for term in set(terms):
                if term not in self.index:
                    self.index[term] = set()
                self.index[term].add(doc_id)

    def get_postings(self, term: str) -> Set[str]:
        processed_terms = self.preprocessor.preprocess(term)
        if not processed_terms:
            return set()
        return self.index.get(processed_terms[0], set()).copy()

    def get_statistics(self) -> Dict[str, int]:
        return {
            "total_documents": len(self.all_doc_ids),
            "vocabulary_size": len(self.index),
            "total_postings": sum(len(p) for p in self.index.values())
        }

    def format_sample_postings(self, sample_terms: List[str], max_docs_shown: int = 8) -> str:
        lines = ["inverted_index_sample = {"]
        for raw_term in sample_terms:
            processed = self.preprocessor.preprocess(raw_term)
            if not processed:
                continue
            term = processed[0]
            postings = sorted(list(self.index.get(term, set())), key=lambda x: int(x[1:]))
            total_matches = len(postings)
            shown = postings[:max_docs_shown]
            formatted_set = ", ".join(f'"{d}"' for d in shown)
            if total_matches > max_docs_shown:
                formatted_set += f", ... (+{total_matches - max_docs_shown} more)"
            lines.append(f'    "{term}": {{{formatted_set}}},  # Total matches: {total_matches}')
        lines.append("}")
        return "\n".join(lines)

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "dataset", "documents.csv")
    df = pd.read_csv(csv_path)
    prep = TextPreprocessor(remove_stopwords=True, normalization="lemmatize")
    inv_idx = InvertedIndex(preprocessor=prep)
    inv_idx.build_index(df)
    stats = inv_idx.get_statistics()

    print("\n" + "═"*80)
    print("INVERTED INDEX")
    print("═"*80 + "\n")
    
    print("Index Statistics:")
    print("┌" + "─"*40 + "┐")
    print(f"│ Total Indexed Documents : {stats['total_documents']:<13}│")
    print(f"│ Unique Vocabulary Terms : {stats['vocabulary_size']:<13}│")
    print(f"│ Total Postings Entries  : {stats['total_postings']:<13}│")
    print("└" + "─"*40 + "┘\n")
    
    print("Sample Postings:")
    print(inv_idx.format_sample_postings(["pizza", "taco", "spicy", "cheap"], max_docs_shown=4))
    print("\n" + "═"*80 + "\n")

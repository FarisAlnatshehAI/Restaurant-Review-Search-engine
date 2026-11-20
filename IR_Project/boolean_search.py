import os
import re
from typing import Dict, Set, Any
import pandas as pd
from preprocessing import TextPreprocessor
from inverted_index import InvertedIndex

class BooleanSearchEngine:
    def __init__(self, inverted_index: InvertedIndex):
        self.inv_index = inverted_index
        self.preprocessor = inverted_index.preprocessor

    def _get_term_set(self, raw_term: str) -> Set[str]:
        processed = self.preprocessor.preprocess(raw_term)
        if not processed: return set()
        result_set = self.inv_index.index.get(processed[0], set()).copy()
        for t in processed[1:]:
            result_set &= self.inv_index.index.get(t, set())
        return result_set

    def search(self, query: str) -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            return {"query": query, "status": "empty_query", "message": "Empty query provided.", "total_matches": 0, "results": []}

        clean_query = query.strip()
        
        and_not_match = re.split(r'\bAND\s+NOT\b', clean_query, maxsplit=1, flags=re.IGNORECASE)
        if len(and_not_match) == 2:
            left_set, right_set = self._get_term_set(and_not_match[0].strip()), self._get_term_set(and_not_match[1].strip())
            return self._format_output(clean_query, f"AND NOT", left_set & (self.inv_index.all_doc_ids - right_set))

        not_match = re.match(r'^NOT\b\s+(.+)$', clean_query, flags=re.IGNORECASE)
        if not_match:
            return self._format_output(clean_query, "NOT", self.inv_index.all_doc_ids - self._get_term_set(not_match.group(1).strip()))

        if re.search(r'\bAND\b', clean_query, flags=re.IGNORECASE):
            sets = [self._get_term_set(p) for p in re.split(r'\bAND\b', clean_query, flags=re.IGNORECASE) if p.strip()]
            return self._format_output(clean_query, "AND", set.intersection(*sets) if sets else set())

        if re.search(r'\bOR\b', clean_query, flags=re.IGNORECASE):
            sets = [self._get_term_set(p) for p in re.split(r'\bOR\b', clean_query, flags=re.IGNORECASE) if p.strip()]
            return self._format_output(clean_query, "OR", set.union(*sets) if sets else set())

        return self._format_output(clean_query, "Implicit AND", self._get_term_set(clean_query))

    def _format_output(self, query: str, operation: str, matched_ids: Set[str]) -> Dict[str, Any]:
        sorted_ids = sorted(list(matched_ids), key=lambda x: int(x[1:]))
        results = [{"doc_id": d, "title": self.inv_index.doc_metadata[d]["title"], "category": self.inv_index.doc_metadata[d]["category"]} for d in sorted_ids]
        return {"query": query, "status": "success", "operation": operation, "total_matches": len(results), "results": results}

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "dataset", "documents.csv")
    df = pd.read_csv(csv_path)
    prep = TextPreprocessor(remove_stopwords=True, normalization="lemmatize")
    inv_idx = InvertedIndex(preprocessor=prep)
    inv_idx.build_index(df)
    bool_eng = BooleanSearchEngine(inv_idx)

    print("\n" + "═"*80)
    print("BOOLEAN SEARCH")
    print("═"*80 + "\n")

    for q in ["hot AND delicious", "pizza OR burger", "NOT terrible", "delicious AND NOT slow"]:
        res = bool_eng.search(q)
        print(f" Query: '{q}'")
        print(f"  Operation: {res['operation']} | Total Matches: {res['total_matches']}")
        print("  Top Results:")
        for doc in res['results'][:3]:
            print(f"     {doc['doc_id']}: {doc['title']} [{doc['category']}]")
        print("─"*60 + "\n")

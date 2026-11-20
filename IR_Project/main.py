import os
import pandas as pd

from prepare_dataset import build_dataset
from preprocessing import TextPreprocessor
from inverted_index import InvertedIndex
from boolean_search import BooleanSearchEngine
from ranked_search import RankedSearchEngine

def print_banner(title: str):
    print("\n" + "=" * 30)
    print(f"  {title}")
    print("=" * 30 + "\n")

def initialize_system():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset", "documents.csv")

    print("Initializing IR System...")
    if not os.path.exists(dataset_path):
        df = build_dataset(output_path=dataset_path, sample_size=1000, random_state=42)
    else:
        df = pd.read_csv(dataset_path)

    preprocessor = TextPreprocessor(remove_stopwords=True, normalization="lemmatize")

    inv_index = InvertedIndex(preprocessor=preprocessor)
    inv_index.build_index(df)
    bool_engine = BooleanSearchEngine(inv_index)

    ranked_engine = RankedSearchEngine(preprocessor=preprocessor)
    ranked_engine.fit_corpus(df)

    print(
        f"System Ready | Documents: {len(df)} | "
        f"Vocabulary: {inv_index.get_statistics()['vocabulary_size']} | "
        f"TF-IDF Shape: {ranked_engine.tfidf_matrix.shape}"
    )
    return bool_engine, ranked_engine

def interactive_console(bool_engine, ranked_engine):
    print_banner(" Restaurant Review Search")
    print("  [1] Boolean Search (AND / OR / NOT)")
    print("  [2] Ranked Search (TF-IDF + Cosine Similarity)")
    print("  [3] Read Full Review (by Doc ID)")
    print("  [q] Quit\n")

    while True:
        choice = input("Select an option (1/2/3/q): ").strip().lower()
        if choice in {'q', 'quit', 'exit'}:
            print("\nExiting. Goodbye!\n")
            break

        elif choice == '1':
            q = input("\nEnter Boolean Query: ")
            res = bool_engine.search(q)

            if res.get("status") == "empty_query":
                print("\n⚠️ Empty query provided. Please enter a valid Boolean query.\n")
            elif not res.get("results"):
                print("\n❌ No matching documents found.\n")
            else:
                print(f"\n=> Found {res['total_matches']} documents (Showing Top 5):")
                for doc in res['results'][:5]:
                    print(f"   - {doc['doc_id']}: {doc['title']}")
                print()

        elif choice == '2':
            q = input("\nEnter Ranked Query: ")
            res = ranked_engine.search(q, top_k=5)

            if res.get("status") == "empty_query":
                print("\n⚠️ Empty query provided. Please enter at least one keyword.\n")
            elif res.get("status") == "no_valid_terms":
                print("\n⚠️ Query contains only stopwords or invalid characters. Try different keywords.\n")
            elif not res.get("results"):
                print("\n❌ No matching documents found.\n")
            else:
                print("\n=> Top 5 Results:")
                for item in res['results']:
                    print(f"  - Rank {item['rank']} | Score: {item['score']:.4f} | {item['doc_id']} | {item['title']}")
                print()

        elif choice == '3':
            doc_id = input("\nEnter Doc ID (e.g., D477): ").strip().upper()
            meta = bool_engine.inv_index.doc_metadata.get(doc_id)
            if meta:
                print("\n" + "-" * 70)
                print(f"Doc ID   : {doc_id}")
                print(f"Category : {meta['category']}")
                print(f"Title    : {meta['title']}")
                print("-" * 70)
                print(f"Full Review:\n{meta['text']}")
                print("-" * 70 + "\n")
            else:
                print(f"\n❌ Document '{doc_id}' not found. Make sure to use format like D1, D477.\n")

        else:
            print("\n⚠️ Invalid option. Please select 1, 2, 3, or q.\n")

if __name__ == "__main__":
    bool_eng, rank_eng = initialize_system()
    interactive_console(bool_eng, rank_eng)

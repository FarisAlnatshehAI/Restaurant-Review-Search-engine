# Restaurant Review Search Engine (Phase 1)

**Course:** Information Retrieval Systems (902333)  
**University:** Al al-Bayt University — Prince Al Hussein bin Abdullah II Faculty of IT  
**Instructor:** Dr. Manar Mizher  

## 1. Project Overview

This project implements the core components of a Python-based Information Retrieval (IR) system for searching restaurant reviews. Users can retrieve relevant customer reviews using two parallel retrieval models operating on the same underlying document collection:
1. **Boolean Search** using a custom-built Inverted Index (`AND`, `OR`, `NOT`, `AND NOT`).
2. **Ranked Search** using TF-IDF vectorization and Cosine Similarity.

## 2. Dataset

- **Source:** [Yelp Review Full (HuggingFace)](https://huggingface.co/datasets/Yelp/yelp_review_full)
- **Sample Size:** 1,000 distinct English restaurant reviews (stratified sampling: 200 reviews per star rating from 1 to 5 stars, `random_state=42` for reproducibility).
- **Cleaning:** Removed empty/missing entries, very short texts (`length <= 25`), and exact duplicates.
- **Fields:** `doc_id` (`D1`–`D1000`), `title` (generated snippet + star rating), `category` (`1-Star Review` to `5-Star Review`), and `text`.

## 3. Project Structure

```text
Restaurant-Review-Search-engine/
├── IR_Project/
│   ├── dataset/
│   │   └── documents.csv          # Generated 1,000-document -can't upload in GitHub because it is more than 25MB-
│   ├── prepare_dataset.py         # Downloads, cleans, and samples the Yelp dataset
│   ├── preprocessing.py           # Tokenization, stopword removal, lemmatization/stemming
│   ├── inverted_index.py          # Dictionary & Set-based Inverted Index
│   ├── boolean_search.py          # Boolean query processor (AND, OR, NOT, AND NOT)
│   ├── ranked_search.py           # TF-IDF vectorization and Cosine Similarity ranking
│   └── main.py                    # Integrated interactive console interface
├── Restaurant_Review_Search_Proposal.pdf
├── requirements.txt
└── README.md
```

## 4. Setup and Exact Run Commands

### Step 1: Clone the repository
```bash
git clone https://github.com/FarisAlnatshehAI/Restaurant-Review-Search-engine.git
cd Restaurant-Review-Search-engine
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the integrated system (Console Interface)
This command automatically generates `dataset/documents.csv` if it does not exist, builds the Inverted Index and TF-IDF matrix, and launches the interactive menu:
```bash
python IR_Project/main.py
```

### Step 4: Run individual modules (for testing & verification)
If `dataset/documents.csv` has not been generated yet, run `prepare_dataset.py` first (all must run in ):
```bash
python IR_Project/prepare_dataset.py
python IR_Project/preprocessing.py
python IR_Project/inverted_index.py
python IR_Project/boolean_search.py
python IR_Project/ranked_search.py
```

## 5. System Pipeline & Methodology

1. **Text Preprocessing (`preprocessing.py`):**
   - Converts text to lowercase and removes non-alphabetic characters.
   - Tokenizes text using NLTK `word_tokenize`.
   - Removes English stopwords (configurable) and single-character tokens.
   - Applies normalization via `WordNetLemmatizer` (default) or `PorterStemmer`.
2. **Inverted Index (`inverted_index.py`):**
   - Maps each unique processed term to a Python `set` of matching `doc_id`s (`Dict[str, Set[str]]`).
3. **Boolean Search (`boolean_search.py`):**
   - `AND`: Set intersection (`&`)
   - `OR`: Set union (`|`)
   - `NOT`: Set complement against the full indexed document set (`all_doc_ids - term_set`)
   - `AND NOT`: Set difference (`left_set & (all_doc_ids - right_set)`)
4. **TF-IDF & Cosine Similarity (`ranked_search.py`):**
   - Fits a single `TfidfVectorizer` on the 1,000-document corpus using the custom `TextPreprocessor`, producing a TF-IDF matrix of shape `(1000, 8601)`.
   - Transforms user queries using the already-fitted vectorizer (`vectorizer.transform([query])`).
   - Computes Cosine Similarity between the query vector and all document vectors, returning the top-k results sorted in descending order.

## 6. Boolean Matching vs. Ranked Matching

- **Boolean Retrieval (Exact Match):** Treats documents and queries as sets of terms. A document either strictly satisfies the logical expression (`AND`, `OR`, `NOT`) or it does not. All returned documents are treated as equally relevant, with no ranking order. This can result in either too many unranked results or zero results if a query is too restrictive.
- **TF-IDF & Cosine Similarity (Ranked Retrieval):** Represents documents and queries as weighted vectors in a multi-dimensional space. Terms that appear frequently in a specific document (high TF) but rarely across the entire corpus (high IDF) receive higher weights. Cosine Similarity measures the angle between the query vector and each document vector, producing a continuous relevance score between `0.0` and `1.0`. This supports partial matches and places the most relevant documents at the top.

## 7. Sample Queries & Verification

### Boolean Queries Tested
- `hot AND delicious`
- `pizza OR burger`
- `NOT terrible`
- `delicious AND NOT slow`

### Ranked Queries Tested
1. `"meatballs Chinese tasty food"`
2. `"spicy burger"`
3. `"fresh sushi rolls"`

#### Sample Ranked Output (`TF-IDF Matrix Shape: (1000, 8601)`)
- **Query:** `"meatballs Chinese tasty food"`
- **Processed Terms:** `['meatball', 'chinese', 'tasty', 'food']`

| Rank | Doc ID | Cosine Score | Title |
| :--- | :--- | :--- | :--- |
| #1 | D798 | 0.4188 | Best Chinese food in Madison by far (5 S... |
| #2 | D477 | 0.2290 | The meatballs are amazing (5 Stars) |
| #3 | D221 | 0.2109 | I was overly excited about this place,..... |

**Explanation of Top-Ranked Result:**  
Document `D798` achieved the highest Cosine Similarity score (`0.4188`) because it strongly matches key query terms—specifically the relatively rare and high-IDF term `"chinese"` along with `"food"`—giving it a higher weighted overlap with the query vector relative to its document length. In comparison, `D477` (`0.2290`) matched the specific term `"meatball"` but lacked other query terms, placing it second.

## 8. Team Members & Role Allocation

| Student ID | Student Name | Assigned Modules / Role |
| :--- | :--- | :--- |
| 2300908018 | فارس جهاد رجب النتشه (Team Leader) | `ranked_search.py`, `main.py` |
| احمد راجي سعيد عميره | 2400902065 | `prepare_dataset.py` |
| مهند محمد يوسف الحارون | 2400902011 | `preprocessing.py` |
| ليث محمود عيسى السلامين | 2300902178 | `inverted_index.py`, `boolean_search.py` |

## 9. References & Acknowledgments

- **Dataset:** Yelp Review Full Dataset — https://huggingface.co/datasets/Yelp/yelp_review_full
- **Libraries:** `pandas`, `numpy`, `nltk`, `scikit-learn`, `datasets`

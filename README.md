# finmatch

## Project Structure
```
finmatch/
├─ data/
│  └─ processed/                 # Preprocessed data and shared dataset splits
│
├─ preprocessing/               # Data cleaning, fund grouping, and label mapping
│
├─ retrieval/
│  ├─ tfidf/                    # TF-IDF-based retrieval
│  ├─ topic/                    # Topic-based retrieval
│  ├─ word_embeddings/          # Word embedding-based retrieval
│  └─ sentence_embeddings/      # Sentence embedding-based retrieval
│
├─ evaluation/                  # Shared evaluation code
├─ results/                     # Experiment results, evaluation metrics, and plots
│
├─ .env                         # Local environment variables and API keys
├─ .env.example                 # Environment variable template
├─ requirements.txt             # Project dependencies
├─ .gitignore
└─ README.md
```

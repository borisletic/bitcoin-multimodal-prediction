# Bitcoin Multimodal Prediction - Quick Start

Sentiment analysis branch for Bitcoin price prediction using BERT + GRU.

## Requirements

- Python 3.9 - 3.11
- NVIDIA GPU (recommended) or CPU
- 8GB+ RAM

## Setup (5 minutes)

### 1. Create Virtual Environment

```bash
# Windows
py -3.11 -m venv venv311
venv311\Scripts\activate

# Linux/Mac
python3.11 -m venv venv311
source venv311/bin/activate
```

### 2. Install Dependencies

```bash
# With GPU (NVIDIA only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# CPU only
pip install -r requirements.txt
```

### 3. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Run Pipeline (~20 min with GPU)

### Option 1: Automated (Recommended)

```bash
python scripts\train_pipeline.py
```

### Option 2: Step by Step

```bash
# 1. Preprocessing
python scripts\preprocessing.py --input data\bitcoin_tweets_10k.csv --output_dir data\processed

# 2. VADER Baseline
python baseline\vader_baseline.py --data data\processed\tweets_labeled.csv --output_dir baseline\vader_results --mode both

# 3. BERT Training
python scripts\sentiment_branch.py --data data\processed\tweets_labeled.csv --output_dir models\bert_sentiment --epochs 3 --batch_size 32

# 4. GRU Embeddings
python models\gru_sentiment_embedding.py --sentiment_features data\processed\sentiment_features_train.csv --bert_model models\bert_sentiment\bert_sentiment_best.pt --window_size 168 --output_dir models\sentiment_embeddings
```

## Results

After running:
- **VADER**: `baseline/vader_results/` 
- **BERT**: `models/bert_sentiment/` 
- **GRU**: `models/sentiment_embeddings/` 

## Troubleshooting

**Import errors?** → Select correct Python interpreter in VS Code (Ctrl+Shift+P → "Python: Select Interpreter")

**CUDA not available?** → Run CPU version or use Google Colab (free GPU)

**Unicode errors?** → All fixed in latest version

## Authors

- Boris Letic (Student 1) - Sentiment Analysis
- Bogdan Ciplic (Student 2) - Price Analysis

## License

MIT License - For educational purposes

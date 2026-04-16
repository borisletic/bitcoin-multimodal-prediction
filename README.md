# Bitcoin Multimodal Prediction - Quick Start

## Requirements
- Python 3.9 - 3.11
- NVIDIA GPU (recommended) or CPU
- 16GB+ RAM

## Setup

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

### 3. Register Jupyter Kernel
```bash
pip install ipykernel
python -m ipykernel install --user --name=venv311 --display-name "Python (venv311)"
```

### 4. Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Run Pipeline (Jupyter Notebooks)

### **Execute notebooks in order:**

1. **`data/01_EDA_and_Data_Preparation.ipynb`** 

2. **`data/02_Preprocessing_and_Features.ipynb`** 

3. **`data/03_VADER_Baseline.ipynb`** 

4. **`data/04_BERT_Finetuning.ipynb`** 

5. **`data/05_BiLSTM_Alternative.ipynb`** 

6. **`data/06_GRU_Embedding_Final_Evaluation.ipynb`** 

### **Then all of Bogdan's notebooks in /notebooks folder**

## Results

After running all notebooks:
- **Preprocessed data**: `data/processed/`
- **Model checkpoints**: `models/`
- **Embeddings**: `results/embeddings/sentiment_embeddings_64dim.npy`


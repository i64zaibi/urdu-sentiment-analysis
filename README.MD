# Urdu Sentiment Analysis using XLM-RoBERTa

An NLP project that classifies Urdu text into **Positive** or **Negative** sentiment using a fine-tuned XLM-RoBERTa-base transformer model.

## Project Overview

- **Model:** XLM-RoBERTa-base (fine-tuned)
- **Dataset:** Urdu Sentiment Analysis — Khan et al. (2020)
- **Dataset Size:** ~10,955 samples
- **Classes:** Positive, Negative (Binary Classification)
- **Framework:** Hugging Face Transformers, PyTorch
- **Web App:** Streamlit

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 82.9% |
| Test F1-Score | 0.829 |
| Negative F1 | 0.82 |
| Positive F1 | 0.83 |

## Project Structure

urdu-sentiment-analysis/
├── app/
│ └── streamlit_app.py
├── model/
│ └── (download model files - see below)
├── notebooks/
│ └── training.ipynb
├── results/
│ ├── confusion_matrix.png
│ └── eda_visualizations.png
├── requirements.txt
├── README.md
└── .gitignore


## How to Get the Model

The trained model is too large for GitHub. Download it from:
[Google Drive Link](https://drive.google.com/drive/folders/1TLo-SV1kKOsY-i9NVQLkZopKsv3U-arN?usp=drive_link)

Place the downloaded files in the `model/` folder.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/urdu-sentiment-analysis.git
cd urdu-sentiment-analysis
```

2. Create virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Download model files and place in `model/` folder

4. Run the application:
```bash
streamlit run app/streamlit_app.py
```

## Training

Training was done in Google Colab with a T4 GPU. The training notebook is in `notebooks/training.ipynb`.

### Training Configuration
- Max Sequence Length: 384 tokens
- Epochs: 3
- Batch Size: 8
- Learning Rate: 2e-5
- Weight Decay: 0.01
- Random Seed: 42

## Preprocessing

Text preprocessing includes:
- URL removal
- Email removal
- HTML tag removal
- At-mention replacement
- Whitespace normalization

The same preprocessing is applied during both training and inference.

## Limitations

1. Binary classification only (no Neutral class)
2. Trained on movie reviews — may not generalize to all domains
3. Roman Urdu not reliably supported
4. Model confidence is a probability estimate, not a guarantee
5. Ambiguous/neutral text defaults to one class since no Neutral label exists

## Technologies

- Python 3.11
- PyTorch
- Hugging Face Transformers
- Streamlit
- Scikit-learn
- Google Colab (Training)

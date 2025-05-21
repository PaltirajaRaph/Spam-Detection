# Multi-Platform Spam Detection System using NLP and Deep Learning

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.7%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)
![Transformers](https://img.shields.io/badge/Transformers-4.10%2B-yellow)

A comprehensive spam message detection system utilizing state-of-the-art natural language processing techniques and deep learning models. The system can analyze and classify text from various sources including emails, SMS messages, and Telegram messages with high accuracy.

## Project Overview

This project implements multiple spam detection models using different machine learning architectures:

1. **LSTM** - A recurrent neural network approach that processes text sequentially
2. **DistilBERT** - A lighter transformer model with efficient performance
3. **RoBERTa** - A high-accuracy transformer model built on BERT architecture

The system is designed to efficiently classify spam across multiple platforms with high precision and accuracy rates.

## Features

- Multi-source and multi-platform spam detection (Email, SMS, Telegram)
- Comprehensive data preprocessing pipeline
- Multiple model architectures for comparison (LSTM, DistilBERT, RoBERTa)
- Detailed performance evaluation metrics
- Data visualization for analysis
- Batch processing capabilities
- Pre-trained models for immediate use

## Dataset

The system utilizes and combines three different datasets:
- **Enron Email Corpus** - Contains 33,345 labeled email messages
- **SMS Spam Collection** - Standard SMS spam dataset
- **Telegram Spam Messages** - Custom collection of Telegram messages

Total combined dataset size: 54,839 unique messages with spam/ham labels.

## Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| RoBERTa | 98.89% | 98.87% | 98.15% | 98.51% |
| DistilBERT | 98.45% | 98.28% | 97.57% | 97.92% |
| LSTM | 97.12% | 96.81% | 96.43% | 96.62% |

## System Architecture

The project implements a comprehensive end-to-end spam detection system:

1. **Data Preprocessing**
   - Text cleaning and normalization
   - Feature extraction
   - Train/validation/test splitting

2. **Model Training**
   - Custom dataset class implementation
   - Tokenization and encoding
   - Fine-tuning pre-trained models
   - Performance monitoring and validation

3. **Evaluation**
   - Accuracy, precision, recall, F1-score metrics
   - Confusion matrix visualization
   - Test set evaluation

4. **Inference**
   - Single message classification
   - Batch processing capabilities
   - Real-time prediction

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- Transformers 4.10+
- pandas
- numpy
- matplotlib
- scikit-learn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spam-detection-nlp.git
cd spam-detection-nlp
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the datasets or use your own data structured with 'text' and 'label' columns.

### Usage

#### Training a Model

```python
from spam_detector import SpamMessageDetector

# Initialize detector with desired model
detector = SpamMessageDetector("distilbert-base-uncased")

# Train the model
detector.train('data/spam_message_train.csv', 'data/spam_message_val.csv', num_epochs=5)

# Save the trained model
detector.save_model('saved_model_path')
```

#### Evaluating Performance

```python
# Evaluate the model on test data
accuracy, precision, recall, f1 = detector.evaluate('data/spam_message_test.csv')
```

#### Making Predictions

```python
# Single message classification
message = "Congratulations! You've won a $1000 gift card. Call 555-123-4567 now!"
is_spam = detector.detect(message)

# Batch processing
messages = [message1, message2, message3]
results = detector.detect(messages)
```

## Model Descriptions

### LSTM Model

The LSTM model provides a lightweight approach with:
- Custom vocabulary building
- Sequential text processing
- Word embedding layer
- Lower computational requirements

### DistilBERT Model

DistilBERT is a distilled version of BERT that retains most of the performance while being lighter and faster:
- 40% smaller than BERT
- 60% faster than BERT
- Preserves 97% of BERT's language understanding capabilities

### RoBERTa Model

RoBERTa (Robustly Optimized BERT Approach) offers the highest accuracy:
- Enhanced training methodology
- Larger training data
- Improved attention mechanism
- State-of-the-art performance

## Future Work

- Integration with email clients and messaging apps
- Deployment as a web service API
- Real-time streaming message classification
- Multi-language support
- User feedback loop for continuous learning

## Acknowledgments

- Hugging Face Transformers library
- PyTorch team
- Creators of the Enron, SMS, and Telegram datasets

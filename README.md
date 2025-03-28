# BERT Text Classification on IMDB Movie Reviews

This project demonstrates how to use a pre-trained BERT model from Hugging Face Transformers to perform **binary sentiment classification** (positive or negative) on movie reviews from the IMDB dataset.

## Features

- Uses `bert-base-uncased` from Hugging Face Transformers
- Binary classification of movie reviews (positive/negative)
- Simple Trainer API from Hugging Face
- Evaluates model performance using accuracy

##  Dependencies

See `requirements.txt` for full list.

##  Model

We use the BERT model for sequence classification:

We use a small sample from the Hugging Face IMDB dataset for quick training and evaluation


Advance code:
What's New in This Script:
Increased training dataset size.

max_length=256 for efficient memory use.

Extra metrics: F1, precision, recall.

EarlyStoppingCallback to avoid overfitting.

gradient_accumulation_steps for better batch simulating.

fp16 mixed-precision training if using GPU.

save_best_model_at_end=True to store only the best version.

# 📊 Utilizing Pre-Trained Models for Sentiment Analysis in Political News Coverage

## 🔍 Overview

This project implements an **entity-level sentiment classification system** for financial and political news headlines using **FinBERT**, a domain-specific language model built on BERT. The model is fine-tuned on the **SEntFiN 1.0 dataset** to classify sentiment into three categories:

* **Positive**
* **Neutral**
* **Negative**

This work was completed as part of an internship project submission and demonstrates the power of **transformer-based architectures** for domain-specific sentiment analysis.

The system consists of:

1. **Dataset Preprocessing** – Cleaning and parsing entity-level annotations.
2. **Model Fine-Tuning** – Adapting FinBERT for sentiment classification.
3. **Evaluation** – Metrics include accuracy, precision, recall, F1-score, and confusion matrix.
4. **Inference** – Testing on unseen headlines.

## 📂 Repository Structure

| SEntFiN.csv                        # Dataset          
│ Sentiment_Analysis.ipynb           # Implementation
│ Report_Sentiment_Analysis.pdf      # Report

## 📜 Dataset

The Dataset is a high-quality dataset of over **10,000 news headlines** annotated with **entity-level sentiment labels**.  

* **10,000+ headlines** with entity-level sentiment labels.
* Columns: `Title`, `Entity`, `Sentiment`.

## 🚀 Features

* Fine-tuned **FinBERT** for sentiment classification.
* High accuracy (**82.97%**) and **82.78% Macro F1**.
* Handles entity-level sentiment detection.
* Fully reproducible on **Google Colab**.
* Confusion matrix visualization for error analysis.

## 🔧 Tech Stack

* Python, PyTorch, Hugging Face Transformers
* Pandas, NumPy, Matplotlib, Scikit-learn
* Google Colab for GPU training


## 📊 Results

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 82.97% |
| Macro F1 | 82.78% |

Example prediction:

Headline: "Government unveils major tax relief package for startups"
Entity: "Government"
Predicted Sentiment: Neutral (96.4% confidence)

## 📘 Report

Read the full report: Report_Sentiment_Analysis.pdf

## 📜 License
This project is for **educational and internship purposes**. Dataset is credited to [SEntFiN](https://huggingface.co/datasets/zeroshot/SEntFiN).

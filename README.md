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

```│
├── data/
│   └── SEntFiN_1.0.csv               
│
├── notebook/
│   └── Sentiment_Analysis.ipynb        
│
├── report/
│   └── Report_Sentiment_Analysis.pdf  
│


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

```
Headline: "Government unveils major tax relief package for startups"
Entity: "Government"
Predicted Sentiment: Neutral (96.4% confidence)
```


## 📘 Report

Read the full report: Report_Sentiment_Analysis.pdf

## 📚 References

1. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.
2. Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
3. Qin, Y. et al. (2023). SEntFiN: A Dataset for Entity-level Sentiment Classification.
4. Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
5. Vaswani, A. et al. (2017). Attention is All You Need.
6. Sun, C. et al. (2019). How to Fine-Tune BERT for Text Classification.
7. Zhang, Y. et al. (2021). Sentiment Analysis in Finance: A Survey.
8. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
9. Yang, Z. et al. (2019). XLNet: Generalized Autoregressive Pretraining.
10. Brown, T. et al. (2020). Language Models are Few-Shot Learners.


## 📜 License
This project is for **educational and internship purposes**. Dataset is credited to [SEntFiN](https://huggingface.co/datasets/zeroshot/SEntFiN).

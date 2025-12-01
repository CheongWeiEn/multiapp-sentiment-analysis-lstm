# Multi-App Sentiment Analysis with LSTM (TensorFlow/Keras)

This project performs **sentiment analysis** on mobile app reviews using **LSTM-based neural networks**.  
It analyses reviews from multiple popular apps (e.g. Spotify, Netflix, TikTok, WhatsApp, Instagram) and classifies them into **Positive, Neutral, or Negative** sentiment.

📌 **Note:**  
This project was completed as part of a **group assignment**, where each team member contributed to data preparation, model experimentation, evaluation, and documentation.

The notebook in this repository contains **my individual contributions, model iterations, and analysis**, extracted and cleaned from the group project.

---

## 🎯 Objective

Given a text review from a mobile app, predict whether the sentiment is:

- **Negative**  
- **Neutral**  
- **Positive**

This helps app developers and product teams:

- Understand user satisfaction  
- Identify common pain points  
- Prioritise feature updates and bug fixes  

---

## 📊 Dataset

- Reviews collected from **Google Play Store** for multiple apps:
  - Spotify  
  - Netflix  
  - TikTok  
  - WhatsApp  
  - Instagram  
- Each review includes:
  - Review text  
  - Rating or sentiment label (mapped to Negative / Neutral / Positive)

*(Dataset loading and preprocessing steps are included in the notebook.)*

---

## 🧼 Text Preprocessing

Key NLP preprocessing steps include:

- Lowercasing  
- Removing URLs & HTML tags  
- Removing punctuation & special characters  
- Tokenisation using **Keras Tokenizer**  
- Converting text to sequences  
- Padding sequences to a uniform length  

Labels are encoded for multi-class classification.

---

## 🧠 Model Architecture (LSTM)

The notebook experiments with multiple LSTM models. A typical architecture includes:

- **Embedding layer** (e.g., vocab size = 10,000, embedding dim = 128)  
- **LSTM layer(s)** (e.g., 32 units)  
- **Dropout** to reduce overfitting  
- Optional **L2 regularisation**  
- Fully-connected Dense layers  
- Final **softmax layer** for 3-class output  

Hyperparameters such as learning rate, optimiser, dropout rate, batch size and number of epochs were tuned during experimentation.

---

## 📈 Model Performance (Example)

Best-performing LSTM model results:

- **Spotify test accuracy:** ~81%  
- **Cross-app generalisation** (trained on one app, tested on other apps):
  - Highest average accuracy across apps: **~61%**
  - Selected as the **group’s final model** for reporting

The notebook includes comparison plots and training/validation curves.

---

## 👥 Group Work Acknowledgement

This project was completed as part of a **collaborative group assignment**.  
Team contributions included:

- Collecting and cleaning review datasets  
- Designing multiple LSTM architectures  
- Running model experiments across different apps  
- Evaluating cross-app generalisation  
- Compiling performance comparisons  
- Preparing written explanations and visualisations  

The notebook in this repository showcases **my personal contributions**, including my model version, analysis, and documentation.

---

## 📂 Repository Structure

```text
multiapp_sentiment_lstm.ipynb
README.md
```

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib / Seaborn
- NLP preprocessing tools

## 💡 Key Learnings
- Text preprocessing for neural-network-based NLP
- Designing and tuning LSTM models
- Cross-application generalisation challenges
- Understanding the differences between training on clean vs noisy reviews
- Collaborative workflow in an ML group project

## 🚀 Future Improvements
- Use pretrained embeddings (GloVe, FastText)
- Try Bi-LSTM, GRU, or Transformer-based models
- Add Attention mechanisms for better interpretability
- Balance classes using oversampling or focal loss
- Deploy the model as a simple web service (Flask/Streamlit)

## 👤 Author
Cheong Wei En
Data Science Student @ Ngee Ann Polytechnic
LinkedIn: https://www.linkedin.com/in/cheong-wei-en-222911303

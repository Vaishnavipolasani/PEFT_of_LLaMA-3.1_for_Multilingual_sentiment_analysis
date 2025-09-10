# Multilingual Sentiment Analysis using LLaMA 3.1–8B with LoRA  

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)  
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)  
![LoRA](https://img.shields.io/badge/LoRA-PEFT-green)  
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)  

---

## 📌 Project Overview  
This project fine-tunes **LLaMA 3.1–8B Instruct** using **LoRA (Low-Rank Adaptation)** to classify sentiment (**Positive / Negative**) in text written in **13 Indian languages**.  

With just **1,000 labeled samples**, the model achieves **97% accuracy**, showing how parameter-efficient fine-tuning can adapt large models for **low-resource multilingual NLP**.  

---

## 🎯 Problem Statement  
Most sentiment analysis models are English-only and fail on Indian languages due to:  
- Script diversity  
- Limited labeled data  
- Computational constraints  

This project solves these challenges by **efficiently fine-tuning a large model** for multilingual sentiment analysis.  

---

## ⚙️ Workflow  
1. **Dataset Preparation**  
   - 1,000 training + 100 test samples across 13 languages  
   - Converted into **instruction–response pairs** in ShareGPT format  

2. **Fine-Tuning**  
   - Base: **LLaMA 3.1–8B Instruct**  
   - **LoRA adapters** on attention & feedforward layers  
   - **4-bit quantization** for memory efficiency  
   - Fine-tuned with **Hugging Face TRL (SFTTrainer)**  

3. **Evaluation**  
   - Metric: **F1 Score**  
   - Achieved **97% accuracy** on test set  

4. **Inference**  
   - Predict sentiment for new sentences  
   - Exported predictions as `submission.csv`  

---

## 🛠️ Technologies Used  
- **LLaMA 3.1–8B Instruct (Meta AI)**  
- **LoRA (Low-Rank Adaptation)**  
- **Unsloth library** for efficient training  
- **Hugging Face Transformers & TRL**  
- **Python, PyTorch, Pandas**  

---

## 🙋 My Contribution  
- Preprocessed and formatted dataset into instruction-response pairs  
- Applied **LoRA fine-tuning** with Unsloth & Hugging Face  
- Trained and evaluated model performance  
- Built inference pipeline for predictions  

---

## 🚀 Use Cases  
- **Multilingual chatbots** for customer support  
- **Social media sentiment analysis** in Indian languages  
- **Market research & opinion mining**  

---

## ⚡ Challenges  
- Small dataset (1k samples)  
- Multiple low-resource languages  
- GPU memory limits  

✅ Solved with **LoRA**, **quantization**, and efficient dataset formatting.  

---

## 🔮 Future Scope  
- Extend to more Indian & global languages  
- Collect larger datasets for improved generalization  
- Deploy model as an **API** for real-time sentiment analysis  

---

## 📊 Results  
| Model                | Accuracy | F1 Score |
|-----------------------|----------|----------|
| LLaMA 3.1–8B + LoRA  | **97%**  | High     |

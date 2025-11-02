# 🌾 Shambani: Intelligent Crop Recommendation System

> **Empowering farmers with data-driven agricultural intelligence**

---

🟢 **Live Demo:** [Shambani – Intelligent Crop Recommender](https://crops-recomendation-system.streamlit.app/)

---

## 🚜 Overview

**Shambani** is a machine learning–powered agricultural advisory platform that recommends the **most suitable crops** for cultivation based on real environmental and soil parameters.  
By analyzing **Nitrogen (N)**, **Phosphorus (P)**, **Potassium (K)**, **temperature**, **humidity**, **pH**, and **rainfall**, it identifies the crops that thrive best under those specific conditions.

Traditional crop selection methods rely heavily on experience and guesswork. Shambani introduces **precision farming intelligence**—making advanced data science accessible to every farmer through a simple and intuitive web interface.

---

## 🎯 Problem Statement

Agricultural productivity is often limited by poor decision-making around crop choice. Farmers may cultivate crops unsuited to their soil or climate, leading to:
- Low yields and wasted inputs (fertilizers, water, and seeds).
- Increased environmental impact.
- Economic losses due to mismatched crop–soil conditions.

The project addresses this challenge by applying **machine learning** to recommend optimal crops for a given environment, thereby improving both **yield quality** and **resource efficiency**.

---

## 🧠 Approach & Methodology

1. **Dataset**
   - Sourced from **Kaggle’s Crop Recommendation Dataset**, which contains ~2,200 samples with features representing key soil nutrients and weather conditions.
   - The dataset is clean, balanced, and representative of typical agricultural data across various crops.

2. **Data Preprocessing**
   - Feature scaling with `StandardScaler` ensures uniform contribution of each parameter to the model.
   - Crop names (categorical target) are encoded numerically using `LabelEncoder`.
   - Exploratory analysis (via Seaborn and Matplotlib) is used to study feature distributions and relationships.

3. **Modeling**
   - **XGBoost Classifier** was selected for its:
     - High performance on structured/tabular data.
     - Robustness to feature interactions.
     - Proven reliability in agricultural data applications.
   - The model predicts the most probable crops suitable for given conditions and ranks them by confidence.

4. **Evaluation**
   - Achieved over **95% accuracy** on held-out test data.
   - Evaluated with confusion matrix and standard classification metrics to confirm generalization.

5. **Deployment**
   - Deployed using **Streamlit** for an interactive and lightweight web interface.
   - The model (`crop_model.json`), scaler, and encoder are stored and loaded dynamically for inference.
   - Farmers can input their soil/environmental data and instantly receive top 3 crop recommendations with confidence percentages.

---

## 🌍 Streamlit WebApp

### Features
- **Interactive Input Form**: Users provide soil and weather parameters.
- **Top-3 Crop Recommendations**: Displayed with confidence percentages.
- **Intuitive Interface**: Designed for accessibility and clarity for agricultural users.
- **Real-time Prediction**: Instant inference powered by the trained XGBoost model.

### Tech Stack
- **Python 3.12+**
- **XGBoost** – model training and inference  
- **Pandas / NumPy / Scikit-learn** – data handling and preprocessing  
- **Matplotlib / Seaborn** – feature visualization  
- **Streamlit** – UI development and deployment  

---

## 📊 Example Prediction

**Input**

| N | P | K | Temperature | Humidity | pH | Rainfall |
|---|---|---|--------------|-----------|----|-----------|
| 90 | 42 | 43 | 20.8 | 82.0 | 6.5 | 200.0 |

**Predicted Crop:** 🌾 *Rice*

The model outputs the top three likely crops, e.g.:

| Crop | Confidence |
|------|-------------|
| Rice | 94.3% |
| Maize | 3.7% |
| Sugarcane | 2.0% |

---

## 🧮 Core Files

| File | Description |
|------|--------------|
| `crop_model.json` | Trained XGBoost model in JSON format |
| `scaler.pkl` | Fitted StandardScaler for input normalization |
| `label_encoder.pkl` | Fitted LabelEncoder for decoding crop names |
| `app.py` | Streamlit app source file |
| `notebook.ipynb` | Jupyter Notebook containing data analysis and model training |

---

## 📚 References & Resources

1. Kaggle – [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)  
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.  
3. Scikit-learn Documentation – [Model Selection & Preprocessing](https://scikit-learn.org/stable/)  
4. Streamlit Documentation – [Building Interactive ML Apps](https://docs.streamlit.io/)  
5. Pandas Documentation – [Data Analysis in Python](https://pandas.pydata.org/docs/)

---

## ⚖️ Disclaimer

**Shambani** is an educational and research-oriented project built to demonstrate the potential of machine learning in sustainable agriculture.  
It should not replace professional agronomic advice. Environmental variations, seed quality, and regional factors must be considered before applying model recommendations in real-world cultivation.

---

## 💡 Vision

Shambani aims to evolve into a **precision farming intelligence platform**, integrating localized weather APIs, soil sensors, and satellite data for region-specific crop optimization — helping farmers transition from intuition-driven to **data-driven agriculture**.

---

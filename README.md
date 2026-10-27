# 📁 Project:26
# 🌴 Date Fruit Variety Classification & Deep Learning Interpretability

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://date-varieties-deep-learning-interpretability.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-f97316.svg)](https://scikit-learn.org/)

An end-to-end, production-grade machine learning and deep learning pipeline designed to automatically classify **7 distinct date fruit varieties** based on high-dimensional morphological, geometric, and color features. This repository features a fully trained Sequential Neural Network (ANN), SHAP-based model interpretability, and a high-contrast glassmorphic Streamlit web application deployed live.

---

## 🚀 Live Demo
Experience the live application here: **[Date Fruit Variety Classifier & Analytics App](https://date-varieties-deep-learning-interpretability.streamlit.app/)**

---

## 📊 Project Overview
Agricultural sorting and quality control require rapid, precise classification of fruit varieties to optimize supply chain efficiency. This project utilizes the **Date Fruit Dataset** (`Date_Fruit_Datasets.xlsx`), consisting of **34 morphological, shape, and color attributes** across **7 commercial date varieties**:
1. **BERHI**
2. **DEGLET**
3. **DOKOL**
4. **IRAQI**
5. **ROTANA**
6. **SAFAVI**
7. **SOGAY**

---

## 🧠 Model Architecture & Performance
* **Model Type:** Sequential Multi-Layer Perceptron (ANN) with staggered Dropout regularization to mitigate overfitting.
* **Preprocessing Pipeline:** Robust feature scaling via Scikit-Learn's `StandardScaler` and stratified three-way train/validation/test splitting to prevent data leakage.
* **Performance:** Achieves robust generalization with **~94% test accuracy**.
* **Interpretability:** Integrated SHAP (`KernelExplainer`) analysis to uncover exact feature contributions (such as `AREA`, `PERIMETER`, `MeanRR`, and `SHAPEFACTOR_1`) driving model decisions.

---

## 💻 Streamlit Web Application Features
The web app is engineered with a modern **glassmorphic dark UI**, high-contrast text visibility, and custom background aesthetics across three interactive tabs:
* **🎯 Prediction Engine:** Input custom morphological measurements, shape factors, and RGB skin color metrics via categorized UI sections to receive real-time class predictions and confidence probability distributions.
* **📈 Exploratory Business Analytics:** Interactive Plotly charts addressing critical agricultural questions:
  * *Q1: Size Grading* (Area vs. Perimeter morphological separation)
  * *Q2: Color Profiles* (Mean Red vs. Mean Green optical skin pigmentation clustering)
  * *Q3: Shape Factor Uniformity* (Geometric consistency across varieties)
* **📖 Variety Knowledge Base:** Comprehensive reference cards detailing the physical characteristics, shapes, and texture profiles of all 7 date fruit classes.

---

## 📂 Repository Structure
```text
├── app.py                               # Main Streamlit web application
├── requirements.txt                     # Project dependencies & exact version pins
├── Date_Fruit_Datasets.xlsx             # Source dataset (34 features, 7 classes)
├── date_fruit_classification_model.keras # Trained Keras neural network weights
└── README.md                            # Project documentation
```

## 🛠️ Local Installation & Setup

1.git clone [https://github.com/M-Nafay-Ali/date-varieties-deep-learning-interpretability.git](https://github.com/M-Nafay-Ali/date-varieties-deep-learning-interpretability.git)
cd date-varieties-deep-learning-interpretability

2.Create and Activate a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3.Install Dependancies
pip install -r requirements.txt

4.Run Streamlit App Locally
streamlit run app.py


## 🌾 Business & Agricultural Impact
1.Automated Quality Inspection: Replaces subjective manual grading with objective, high-speed neural network inference.

2.Data-Driven Sorting: Empowers sorting machinery with multi-feature fusion (combining size, shape factors, and optical color histograms) rather than simple linear thresholding.

## 📞 Contact Information:-
* **Email:-**[englandengland271@gmail.com]
* **Linkedin:-**[https://www.linkedin.com/in/mohammed-nafay-ali-16519138a?utm_source=share_via&utm_content=profile&utm_medium=member_android]
* **GitHub:-**[https://github.com/M-Nafay-Ali]


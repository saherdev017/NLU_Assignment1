``markdown
# 📰 NLU Text Classification: Sports vs. Politics

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-Feature%20Engineering-success.svg)]()
[![IIT Jodhpur](https://img.shields.io/badge/Institution-IIT%20Jodhpur-800000.svg)]()

> A robust Natural Language Understanding (NLU) pipeline designed to classify journalistic text into binary categories using from-scratch mathematical feature engineering and classical machine learning algorithms.

🌐 **[View the Live Project Webpage Here](https://saherdev017.github.io/NLU_Assignment1/)**

---

## 📖 Project Overview
This project tackles a foundational problem in Natural Language Processing: automated text categorization. Using a filtered dataset of 10,000 news articles, this pipeline processes raw text, generates highly optimized mathematical vectors **from scratch**, and trains multiple classical machine learning models to accurately distinguish between "Sports" and "Politics" reporting.

### 🚀 Key Technical Highlights
* **From-Scratch Feature Engineering:** Bypassed standard library vectorizers to mathematically build Bag of Words (BoW), TF-IDF, and N-Gram extractors from the ground up using core Python.
* **Sparse Matrix Optimization:** Compressed highly dimensional text vectors ($8000 \times 15000$) into `scipy.sparse` CSR matrices, drastically reducing memory footprint and resulting in near-instantaneous model training.
* **Advanced Visual EDA:** Features normalized percentage confusion matrices, custom-colored circular Word Clouds, and Lollipop charts to visually explain vocabulary divergence.
* **Comprehensive Evaluation:** Evaluates 9 distinct combinations of feature representations and machine learning algorithms.

---

## 📊 Methodology

### 1. Data Preprocessing
* Ingested data from the Kaggle News Category Dataset.
* Concatenated `headline` and `short_description` for maximized contextual density.
* Applied regex-based cleaning to strip URLs, non-alphabetical characters, and excessive whitespace.
* Removed a custom-defined set of 90 standard English stop words.

### 2. Feature Representation (Custom Implementations)
* **Bag of Words (BoW):** Unigram frequency counting.
* **TF-IDF:** Term Frequency-Inverse Document Frequency with strict $L_2$ Normalization.
* **N-Grams:** Expanded tokenization to include contextual Bigrams ($word_i\_word_{i+1}$).

### 3. Machine Learning Models
* **Multinomial Naive Bayes** (with Laplace smoothing)
* **Logistic Regression** (Optimized via SGD)
* **Linear Support Vector Machine** ($L_2$ Regularized Hinge Loss)

---

## 🏆 Results & Performance

The pipeline achieved exceptional accuracy, driven by the strong orthogonal semantic separation between political and sports terminology. **Logistic Regression paired with Unigram TF-IDF** emerged as the optimal architecture.

| Feature Strategy | Multinomial NB | Logistic Regression | Linear SVM |
| :--- | :---: | :---: | :---: |
| **Bag of Words** | 97.85% | 98.00% | 97.45% |
| **TF-IDF** | 96.85% | **98.15%** | 97.95% |
| **N-Grams (Bigrams)** | 96.70% | 98.10% | 98.05% |

---

## 💻 Repository Structure

```text
├── images/                           # Generated EDA and Evaluation graphs
│   ├── tfidf_lollipop.png            
│   ├── wordclouds.jpg                
│   ├── conf_matrix.png               
│   └── comparison_bar.png            
├── B23CS1059_prob4_dataset.csv       # Cleaned, binary-filtered 10k dataset
├── index.html                        # Source code for the live GitHub Page
├── NLU_Classification_Notebook.ipynb # Main Jupyter Notebook with all logic
├── Report_NLU_SaherDev.pdf           # 5-page IEEE formatted academic report
└── README.md                         # Project documentation

```

---

## ⚙️ Usage & Installation

To run this pipeline locally on your machine:

1. **Clone the repository:**
```bash
git clone https://github.com/saherdev017/NLU_Assignment1
cd NLU_Assignment1

```


2. **Install the required dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy wordcloud

```


3. **Run the Notebook:**
Open `NLU_Classification_Notebook.ipynb` in Jupyter or Google Colab and run the cells sequentially.

---

## 👤 Author

**Saher Dev** B.Tech Computer Science and Engineering, Class of 2027

Indian Institute of Technology (IIT) Jodhpur

*Built for NLU Coursework*

```

***

```

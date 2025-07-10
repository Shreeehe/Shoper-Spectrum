# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations

Hi there! 👋 This project explores customer shopping behavior in an e-commerce setting, aiming to **segment customers** and **recommend products** they may love 💛

---

## 🎯 Project Objectives

- Segment customers using **RFM Analysis** (Recency, Frequency, Monetary).
- Use **unsupervised machine learning** to create clusters.
- Build a **recommendation system** based on purchase patterns.
- Deploy an interactive app using **Streamlit**.

---

## 🧠 Techniques Used

- 🧼 Data Cleaning & Feature Engineering  
- 📊 Exploratory Data Analysis (EDA)  
- 🎯 RFM Segmentation  
- 🤖 KMeans Clustering  
- 🤝 Collaborative Filtering  
- 🧠 Cosine Similarity  
- 🌐 Streamlit App Deployment  

---

## 📦 Dataset Info

> You can find the dataset [here](https://drive.google.com/file/d/1rzRwxm_CJxcRzfoo9Ix37A2JTlMummY-/view?usp=sharing)

**Columns include:**

- `InvoiceNo` – Order number  
- `StockCode` – Product ID  
- `Description` – Product name  
- `Quantity` – Number of items bought  
- `InvoiceDate` – Transaction date  
- `UnitPrice` – Price per unit  
- `CustomerID` – Customer unique ID  
- `Country` – Customer's country  

---

## 🧩 Modules in the Streamlit App

### 1️⃣ Product Recommendation  
💬 Enter a product name → get **5 similar products** recommended using item-based collaborative filtering.

### 2️⃣ Customer Segmentation  
🔢 Enter Recency, Frequency, and Monetary values → predict the customer's segment such as:  
- 💎 High-Value  
- 👍 Regular  
- 😌 Occasional  
- 😟 At-Risk  

---
## 🚀 How to Run

```bash
# Install requirements
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

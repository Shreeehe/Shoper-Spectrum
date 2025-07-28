import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load the KMeans model here
with open("rfm_kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load RFM CSV
rfm_df = pd.read_csv("rfm_data.csv")

# ✅ Scale it first
scaler = MinMaxScaler()
scaled_rfm = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

# ✅ Predict
clusters = model.predict(scaled_rfm)

# Add back to dataframe
rfm_df['Cluster'] = clusters


# Cluster names
cluster_names = {
    0: "High-Value",
    1: "At-Risk",
    2: "Occasional",
    3: "Regular"
}

st.set_page_config(page_title="Shopper Spectrum", layout="wide")
st.title("🛒 Shopper Spectrum: Customer Segmentation & Product Recommendation")

# Tabs
tab1, tab2 = st.tabs(["📈 Customer Segmentation", "🎯 Product Recommendation"])

# ====================== 📈 Customer Segmentation Tab ========================
with tab1:
    st.header("Customer Segmentation (RFM Clustering)")
    rfm_file = st.file_uploader("Upload RFM CSV (with Recency, Frequency, Monetary)", type=["csv"], key="rfm")

    if rfm_file is not None:
        rfm_df = pd.read_csv(rfm_file)

        if {'Recency', 'Frequency', 'Monetary', 'CustomerID'}.issubset(rfm_df.columns):

            # Scale RFM
            scaler = MinMaxScaler()
            scaled_rfm = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

            # Predict clusters
            clusters = model.predict(scaled_rfm)
            rfm_df['Cluster'] = clusters
            rfm_df['Segment'] = rfm_df['Cluster'].map(cluster_names)

            st.subheader("📋 Customer Segmentation Table")
            st.dataframe(rfm_df.head(10))

            st.subheader("📊 Segment Distribution")
            seg_counts = rfm_df['Segment'].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%', startangle=140)
            ax1.axis("equal")
            st.pyplot(fig1)

            st.subheader("📉 Recency vs Monetary Scatter")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=rfm_df, x="Recency", y="Monetary", hue="Segment", palette="Set2", ax=ax2)
            st.pyplot(fig2)

        else:
            st.error("❗ CSV must contain: Recency, Frequency, Monetary, CustomerID")

# ====================== 🎯 Product Recommendation Tab ========================
with tab2:
    st.header("Item-Based Collaborative Filtering")

    retail_file = st.file_uploader("Upload Cleaned E-Commerce CSV", type=["csv"], key="retail")

    if retail_file is not None:
        df = pd.read_csv(retail_file)

        if {'CustomerID', 'Description', 'Quantity'}.issubset(df.columns):
            pivot = df.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum', fill_value=0)
            item_sim = cosine_similarity(pivot.T)
            sim_df = pd.DataFrame(item_sim, index=pivot.columns, columns=pivot.columns)

            st.subheader("🔍 Enter Product Name")
            product_name = st.text_input("Enter exact product name to get similar items")

            def recommend_items(name, sim_df, n=5):
                if name not in sim_df.columns:
                    return [f"❌ '{name}' not found"]
                return sim_df[name].sort_values(ascending=False).iloc[1:n+1].index.tolist()

            if product_name:
                recs = recommend_items(product_name, sim_df)
                st.subheader("🛍️ Top 5 Similar Products")
                for i, item in enumerate(recs, start=1):
                    st.markdown(f"**{i}. {item}**")
        else:
            st.error("❗ CSV must contain: CustomerID, Description, Quantity")

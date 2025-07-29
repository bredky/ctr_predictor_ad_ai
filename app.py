import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import joblib
import openai
import json
from sklearn.metrics.pairwise import cosine_similarity
from prompts import generate_pros_cons_prompt
from dotenv import load_dotenv
load_dotenv(".env.local")
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



from model.train_model import load_and_process_data, train_model
from clip_model.clip_extractor import extract_clip_features

st.set_page_config(page_title="CTR Predictor", layout="wide")
st.title(" Ad Creative CTR Predictor")

# Sidebar
st.sidebar.title(" Upload Campaign Data")
excel_file = st.sidebar.file_uploader("Upload your campaign metrics Excel file", type=["xlsx"])

image_folder = st.sidebar.text_input("Path to your image folder", value="images/")

# Tabs
tab1, tab2 = st.tabs([" View Predictions", " Predict New Ad"])

# ----------------------------- #
# TAB 1: View Predictions
# ----------------------------- #
with tab1:
    if excel_file and image_folder:
        st.info("Processing data and extracting image features...")
        raw_df = pd.read_excel(excel_file)
        raw_df.columns = [col.strip() for col in raw_df.columns]

        grouped = raw_df.groupby("Creative", as_index=False).agg({
            "Impressions": "sum",
            "Clicks": "sum"
        })

        grouped["CTR"] = grouped["Clicks"] / grouped["Impressions"]
        grouped = grouped.rename(columns={"Creative": "campaign_name"})

        os.makedirs("data", exist_ok=True)
        processed_path = "data/clean_grouped_metrics.xlsx"
        grouped.to_excel(processed_path, index=False)
        st.markdown("### grouped metrics")
        st.dataframe(grouped)

        X = []
        y = []
        valid_rows = []

        for _, row in grouped.iterrows():
            image_path = os.path.join(image_folder, f"{row['campaign_name']}.jpg")
            if os.path.exists(image_path):
                try:
                    features = extract_clip_features(image_path)
                    X.append(features)
                    y.append(row["CTR"])
                    valid_rows.append(row)
                except Exception as e:
                    print(f"Skipping {row['campaign_name']} due to error: {e}")
            else:
                print(f"Image not found for {row['campaign_name']}")

        df = pd.DataFrame(valid_rows)


        if len(X) == 0:
            st.error("No valid images found matching the campaign names.")
        else:
            model, raw_preds, r2 = train_model(X, y)

            
            preds = np.clip(raw_preds, 0, 1)

            df["Predicted CTR"] = np.round(preds, 4)
            df["Actual CTR"] = np.round(y, 4)
            df["CTR Error"] = np.round(np.abs(df["Predicted CTR"] - df["Actual CTR"]), 4)

            
            df["Image Path"] = df["campaign_name"].apply(lambda x: os.path.join(image_folder, f"{x}.jpg"))

            st.success(f"Model trained! R² Score: **{r2:.4f}**")

            st.subheader(" CTR Predictions")

            st.markdown("### Actual vs Predicted CTR")
            chart_df = df[["campaign_name", "Actual CTR", "Predicted CTR"]].set_index("campaign_name")
            st.bar_chart(chart_df)
            for _, row in df.iterrows():
                col1, col2 = st.columns([1, 3])
                image_path = os.path.join(image_folder, f"{row['campaign_name']}.jpg")

                if os.path.exists(image_path):
                    col1.image(Image.open(image_path), width=150)

                col2.markdown(f"""
                **Campaign:** `{row['campaign_name']}`  
                -  **Actual CTR:** `{row['CTR']:.4f}`  
                - **Predicted CTR:** `{row['Predicted CTR']:.4f}`  
                - **CTR Error:** `{row['CTR Error']:.4f}`
                """)
                st.markdown("---")

with tab2:
    st.subheader("Upload a New Ad Image")
    uploaded_image = st.file_uploader("Upload JPG or PNG", type=["jpg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Creative", use_column_width=True)

        try:
            # Extract CLIP features
            temp_path = "temp_uploaded.jpg"
            image.save(temp_path)
            features = extract_clip_features(temp_path)

            if "model" not in locals():
                st.warning(" Please train a model first in Tab 1.")
            else:
                pred_ctr = float(np.clip(model.predict([features])[0], 0, 1))
                st.success(f" Predicted CTR: **{pred_ctr:.4f}**")

                openai.api_key = os.getenv("OPENAI_API_KEY")
                prompt = generate_pros_cons_prompt(pred_ctr)

                from utils.vision import encode_image_to_base64

                # Encode image for GPT-4-Vision
                base64_image = encode_image_to_base64(temp_path)

                with st.spinner("Analyzing ad..."):
                    chat_completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": base64_image,
                                            "detail": "high"
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": f"""Analyze this ad creative based on the following criteria:
                - Visual appeal and design clarity
                - Message clarity and strength of call-to-action
                - Storytelling or outcome demonstration
                - Use of trust-building elements (reviews, press)
                - Brand visibility and identity
                - Relevance to audience or timing
                - Alignment with CTR and conversion performance

                The predicted CTR is {pred_ctr:.4f}.

                Return 2 bullet points for:
                - Pros (what works)
                - Cons (what could be improved)
                """
                                    }
                                ]
                            }
                        ],
                        max_tokens=500
                    )

                feedback = chat_completion.choices[0].message.content
                st.markdown("### 🤖 GPT-4 Vision Feedback")
                st.markdown(feedback)


                # --- Similarity Search ---
                st.markdown("###  Top 3 Similar Ads")
                db = np.load("model_store.npz", allow_pickle=True)["metadata"]
                db = db.tolist()

                emb_matrix = np.array([entry["embedding"] for entry in db])
                sim_scores = cosine_similarity([features], emb_matrix)[0]
                top_idxs = sim_scores.argsort()[::-1][:3]

                for idx in top_idxs:
                    entry = db[idx]
                    col1, col2 = st.columns([1, 3])
                    if os.path.exists(entry["image_path"]):
                        col1.image(entry["image_path"], width=150)
                    col2.markdown(f"""
                    **{entry['campaign_name']}**  
                    -  Predicted CTR: `{entry['predicted_ctr']:.4f}`  
                    - Actual CTR: `{entry['actual_ctr']:.4f}`
                    """)
                    st.markdown("---")

        except Exception as e:
            st.error(f" Error processing image: {e}")

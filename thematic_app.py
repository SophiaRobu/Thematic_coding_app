import streamlit as st
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from nltk.corpus import stopwords
import nltk
from hdbscan import HDBSCAN

st.set_page_config(page_title="Thematic Coding Tool", layout="wide")

def login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([3, 2, 3])

        with col2:
            st.markdown("### 🔐 Login Required")
            password = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Enter password")

            if password == "Ech0istaTool1":
                st.session_state.authenticated = True
                st.rerun()  
            elif password != "":
                st.error("Incorrect password. Please try again.")
            else:
                st.info("Please enter the password to access the app.")

    return st.session_state.authenticated


# 🔐 Only run the rest of the app if authenticated
if not login():
    st.stop()


st.title("🧠 Thematic Coding with Echo Research")
st.write("Upload your Excel file and automatically assign themes using AI.")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel (.xlsx) file", type=["xlsx"])

if uploaded_file:
    try:
        # Load file
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()

        text_col = st.selectbox("Select the column containing text:", df.columns)

        if st.button("Generate Themes"):
            with st.spinner("Processing..."):

                # Clean text column
                df[text_col] = df[text_col].astype(str).str.strip()
                df = df[df[text_col] != ""].drop_duplicates()

                docs = df[text_col].tolist()

                # Run BERTopic
                umap_model = UMAP(random_state=75)

                hdbscan_model = HDBSCAN(min_cluster_size=5, prediction_data=True)

                # BERTopic using both
                topic_model = BERTopic(
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    calculate_probabilities=False,
                    verbose=False
                )

                topics, _ = topic_model.fit_transform(docs)

                nltk.download('stopwords', quiet=True)
                # Assign topics
                df['Assigned_Topic'] = topics

                stop_words = set(stopwords.words('english'))

            
                topic_info = topic_model.get_topic_info()
                topic_keywords = {
                    row['Topic']: ", ".join([word for word in row['Representation'] if word.lower() not in stop_words][:5])
                    for _, row in topic_info.iterrows()
                    if row['Topic'] != -1
                }
                topic_keywords[-1] = "Uncategorised"

                df['Theme_Label'] = df['Assigned_Topic'].map(topic_keywords)

                # Show results
                st.success("Themes assigned!")
                st.markdown("**Here are the first 5 rows. To view all responses, please download the CSV file below.**")
                st.write(df[['Assigned_Topic', 'Theme_Label', text_col]].head(5))

                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results as CSV", csv, "thematic_output.csv", "text/csv")

                # Show actual topic counts (your truth source)
                st.subheader("📊 Topic Frequency in Your Data")
                topic_counts = df['Assigned_Topic'].value_counts().reset_index()
                topic_counts.columns = ['Topic', 'Count']
                topic_counts['Label'] = topic_counts['Topic'].map(topic_keywords)
                st.dataframe(topic_counts)

                topic_info_raw = topic_model.get_topic_info()

                # Create cleaned top words per topic
                cleaned_rep = []
                for _, row in topic_info_raw.iterrows():
                    if row['Topic'] == -1:
                        cleaned_rep.append("Uncategorised")
                    else:
                        # Remove stopwords and non-alphabetic tokens
                        filtered_words = [w for w in row['Representation'] if w.lower() not in stop_words and w.isalpha()]
                        cleaned_rep.append(", ".join(filtered_words[:5]))

                # Create a new DataFrame with the cleaned "Top Words"
                topic_info_cleaned = topic_info_raw.copy()
                topic_info_cleaned['Top Words'] = cleaned_rep

                # Drop unnecessary columns
                topic_info_cleaned = topic_info_cleaned.drop(columns=['Name', 'Representation'])

                topic_info_cleaned = topic_info_cleaned.rename(columns={"Representative_Docs": "Quotes"})

                # Reorder columns so 'Top Words' comes before 'Representative_Docs'
                column_order = ['Topic', 'Count', 'Top Words', 'Quotes']
                topic_info_cleaned = topic_info_cleaned[column_order]

                # Sort and display
                topic_info_cleaned = topic_info_cleaned.sort_values(by='Count', ascending=False)
                st.subheader("📋 Summary Overview Details")

                # 1. Prepare quotes with line breaks
                topic_info_cleaned['Quotes'] = topic_info_cleaned['Quotes'].apply(
                    lambda quotes: "\n\n".join([q for q in quotes if q.strip()]) if isinstance(quotes, list) else quotes
                )

                # 2. Inject CSS for better spacing (optional, helps with some themes)
                st.markdown("""
                    <style>
                    div[data-testid="stDataEditor"] textarea {
                        white-space: pre-wrap;
                        overflow-wrap: break-word;
                        line-height: 1.6;
                        padding-top: 10px;
                        padding-bottom: 10px;
                    }
                    </style>
                """, unsafe_allow_html=True)

                # 3. Display
                st.data_editor(topic_info_cleaned, use_container_width=True, height=800)



    except Exception as e:
        st.error(f"Something went wrong: {e}")

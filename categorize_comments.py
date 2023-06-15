import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import base64
from io import BytesIO

# Specify download directory for NLTK data
nltk.download('stopwords', download_dir='/home/appuser/nltk_data')
nltk.download('vader_lexicon', download_dir='/home/appuser/nltk_data')
nltk.download('punkt', download_dir='/home/appuser/nltk_data', quiet=True)  # Add 'quiet=True' to suppress NLTK download messages
nltk.data.path.append('/home/appuser/nltk_data')

# Initialize BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Create a dictionary to store precomputed embeddings
keyword_embeddings = {}

# Preprocessing function
def preprocess_text(text):
    # Remove unnecessary characters
    text = str(text)  # Convert to string
    text = text.strip()  # Remove leading/trailing whitespaces

    # Convert to lowercase
    text = text.lower()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    preprocessed_text = ' '.join(filtered_text)

    return preprocessed_text

# Perform sentiment analysis
def perform_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    return compound_score

# Compute semantic similarity
def compute_semantic_similarity(keyword_embedding, comment_embedding):
    similarity = cosine_similarity([keyword_embedding], [comment_embedding])
    return similarity[0][0]

# Streamlit interface
st.title("Feedback Categorization")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# Select the column containing the comments
comment_column = None
if uploaded_file is not None:
    # Read customer feedback from uploaded file
    csv_data = uploaded_file.read()
    feedback_data = pd.read_csv(BytesIO(csv_data))
    comment_column = st.selectbox("Select the column containing the comments", feedback_data.columns.tolist())
    process_button = st.button("Process Feedback")

# Define keyword-based categories
default_categories = {
    'Website Usability': ['user experience', 'navigation', 'interface', 'design', 'loading speed', 'search functionality', 'mobile responsiveness', 'compatibility', 'site organization', 'site performance', 'page loading time', 'difficult to navigate', 'poorly organized', 'difficult to find information', 'not user-friendly', 'hard to use'],
    'Product Quality': ['quality', 'defects', 'durability', 'reliability', 'performance', 'features', 'accuracy', 'packaging', 'product condition'],
    'Shipping and Delivery': ['delivery', 'shipping', 'shipping time', 'tracking', 'package condition', 'courier', 'fulfillment', 'damaged during shipping', 'order tracking'],
    'Customer Support': ['customer support', 'support service', 'support response time', 'communication', 'issue resolution', 'knowledgeability', 'helpfulness', 'replacement', 'refund', 'customer service experience'],
    'Order Processing': ['order', 'payment', 'checkout', 'transaction', 'billing', 'stock availability', 'order confirmation', 'cancellation', 'order customization', 'order accuracy'],
    'Product Selection': ['product customization', 'customization difficulties', 'product variety', 'product options', 'finding the right product', 'product suitability'],
    'Price and Value': ['pricing', 'value for money', 'discounts', 'promotions', 'affordability', 'price discrepancy', 'price competitiveness'],
    'Product Information': ['specifications', 'descriptions', 'images', 'product details', 'accurate information', 'misleading information', 'product data'],
    'Returns and Refunds': ['returns', 'refunds', 'return policy', 'refund process', 'return condition', 'return shipping', 'return authorization'],
    'Warranty and Support': ['warranty', 'technical support', 'repair', 'technical assistance', 'warranty claim', 'product support'],
    'Website Security': ['security', 'privacy', 'data protection', 'secure checkout', 'account security', 'payment security'],
    'Website Performance': ['uptime', 'site speed', 'loading time', 'server stability', 'site crashes', 'website availability'],
    'Accessibility': ['accessibility', 'inclusive design', 'special needs', 'assistive technologies', 'screen reader compatibility', 'website usability for disabled users'],
    'Unwanted Emails': ['spam emails', 'email subscriptions', 'unsubscribe', 'email preferences', 'inbox management', 'email marketing'],
}
# Edit categories and keywords
st.sidebar.header("Edit Categories")
categories = {}
for category, keywords in default_categories.items():
    category_name = st.sidebar.text_input(f"{category} Category", value=category)
    category_keywords = st.sidebar.text_area(f"Keywords for {category}", value="\n".join(keywords))
    categories[category_name] = category_keywords.split("\n")

st.sidebar.subheader("Add or Modify Categories")
new_category_name = st.sidebar.text_input("New Category Name")
new_category_keywords = st.sidebar.text_area(f"Keywords for {new_category_name}")
if new_category_name and new_category_keywords:
    categories[new_category_name] = new_category_keywords.split("\n")

if comment_column is not None and process_button:
    # Initialize lists to store categorized comments and sentiments
    categorized_comments = []
    sentiments = []

# Pre-compute embeddings for all keywords
for category_name, keywords in categories.items():
    for keyword in keywords:
        keyword_embeddings[keyword] = model.encode([keyword])[0]

if comment_column is not None and process_button:
    # Initialize lists to store categorized comments and sentiments
    categorized_comments = []
    sentiments = []

    # Process each comment
    with st.spinner('Processing feedback...'):
        for index, row in feedback_data.iterrows():
            preprocessed_comment = preprocess_text(row[comment_column])
            comment_embedding = model.encode([preprocessed_comment])[0]  # Compute the comment embedding once
            sentiment_score = perform_sentiment_analysis(preprocessed_comment)

            category = 'Other'
            sub_category = 'Other'
            best_match_score = float('-inf')  # Initialized to negative infinity

            # Tokenize the preprocessed_comment
            tokens = word_tokenize(preprocessed_comment)

            for main_category, keywords in categories.items():
                for keyword in keywords:
                    keyword_embedding = keyword_embeddings[keyword]  # Use the precomputed keyword embedding
                    similarity_score = compute_semantic_similarity(keyword_embedding, comment_embedding)
                    # If similarity_score equals best_match_score, we pick the first match.
                    # If similarity_score > best_match_score, we update best_match.
                    if similarity_score >= best_match_score:
                        category = main_category
                        sub_category = keyword
                        best_match_score = similarity_score

            row_extended = row.tolist() + [preprocessed_comment, category, sub_category, sentiment_score]
            categorized_comments.append(row_extended)
            sentiments.append(sentiment_score)

    # Generate trends and insights
    st.success('Done!')
    existing_columns = feedback_data.columns.tolist()
    additional_columns = [comment_column, 'Category', 'Sub-Category', 'Sentiment']
    headers = existing_columns + additional_columns

    # Create a new DataFrame with extended columns
    trends_data = pd.DataFrame(categorized_comments, columns=headers)

    # Rename duplicate column names
    trends_data = trends_data.loc[:, ~trends_data.columns.duplicated()]
    duplicate_columns = set([col for col in trends_data.columns if trends_data.columns.tolist().count(col) > 1])
    for column in duplicate_columns:
        column_indices = [i for i, col in enumerate(trends_data.columns) if col == column]
        for i, idx in enumerate(column_indices[1:], start=1):
            trends_data.columns.values[idx] = f"{column}_{i}"

    # Display trends and insights
    st.title("Feedback Trends and Insights")
    st.dataframe(trends_data)

    # Download Excel file
    st.markdown("""
    ### Download Excel File
    """)

    excel_file = BytesIO()
    trends_data.to_excel(excel_file, index=False, engine='openpyxl')
    excel_file.seek(0)
    b64 = base64.b64encode(excel_file.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="feedback_trends.xlsx">Download Excel File</a>'
    st.markdown(href, unsafe_allow_html=True)

import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# Initialize the SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Create sample healthcare dataset
def create_sample_data():
    return pd.DataFrame({
        'condition': [
            'Heart attack',
            'Myocardial infarction',
            'Cardiac arrest',
            'High blood pressure',
            'Hypertension',
            'Elevated BP',
            'Diabetes type 2',
            'Type 2 diabetes mellitus',
            'Adult onset diabetes',
            'Common cold',
            'Upper respiratory infection',
            'Rhinovirus infection'
        ],
        'description': [
            'Blockage of blood flow to the heart muscle',
            'Acute damage to heart muscle due to lack of blood flow',
            'Sudden stopping of heart function',
            'Condition where blood pressure is consistently too high',
            'Medical condition with elevated arterial pressure',
            'Blood pressure readings above normal range',
            'Metabolic disorder affecting blood sugar levels',
            'Chronic condition affecting glucose metabolism',
            'Diabetes developing in adulthood',
            'Viral infection of upper respiratory tract',
            'Infection affecting nose and throat',
            'Viral infection causing runny nose and congestion'
        ]
    })

def keyword_search(query, data, column):
    """Simple keyword search implementation"""
    return data[data[column].str.contains(query, case=False)].index.tolist()

def vector_search(query, data, model, column):
    """Dense vector search implementation"""
    # Encode query and all descriptions
    query_embedding = model.encode([query])
    doc_embeddings = model.encode(data[column].tolist())
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Return indices sorted by similarity
    return sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)

def main():
    st.set_page_config(page_title="Healthcare Search Comparison", layout="wide")
    
    st.title("Healthcare Search Comparison: Keyword vs Dense Vector Search")
    
    # Initialize data and model
    model = load_model()
    data = create_sample_data()
    
    # Sidebar for search input
    st.sidebar.header("Search Settings")
    search_query = st.sidebar.text_input("Enter your search query:", "heart problem")
    search_column = st.sidebar.selectbox("Search in:", ['condition', 'description'])
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Keyword Search Results")
        keyword_results = keyword_search(search_query, data, search_column)
        
        if keyword_results:
            results_df = data.iloc[keyword_results].copy()
            results_df['Match Type'] = 'Exact Match'
            st.dataframe(results_df)
            
            # Visualization for keyword search
            fig_keyword = go.Figure(data=[go.Table(
                header=dict(values=['Condition', 'Description', 'Match Type'],
                          fill_color='paleturquoise',
                          align='left'),
                cells=dict(values=[results_df['condition'], 
                                 results_df['description'],
                                 results_df['Match Type']],
                          fill_color='lavender',
                          align='left'))
            ])
            st.plotly_chart(fig_keyword)
        else:
            st.write("No exact matches found")

    with col2:
        st.header("Dense Vector Search Results")
        vector_results = vector_search(search_query, data, model, search_column)
        
        # Get similarities for visualization
        query_embedding = model.encode([search_query])
        doc_embeddings = model.encode(data[search_column].tolist())
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Create results dataframe with similarities
        results_df = data.iloc[vector_results].copy()
        results_df['Similarity Score'] = similarities[vector_results]
        results_df = results_df[results_df['Similarity Score'] > 0.3]  # Filter low similarity scores
        
        if not results_df.empty:
            st.dataframe(results_df)
            
            # Bar chart for similarity scores
            fig_vector = px.bar(
                results_df,
                y='condition',
                x='Similarity Score',
                orientation='h',
                title='Semantic Similarity Scores',
                color='Similarity Score',
                color_continuous_scale='Viridis'
            )
            fig_vector.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_vector)
        else:
            st.write("No semantic matches found")
    
    # Explanation section
    st.markdown("""
    ### How it works
    
    #### Keyword Search
    - Looks for exact matches of the search terms
    - Case-insensitive matching
    - Limited to finding exact text patterns
    
    #### Dense Vector Search
    - Converts text into high-dimensional vectors
    - Measures semantic similarity between query and documents
    - Can find related terms and synonyms
    - Shows similarity scores for each match
    
    Try searching for terms like:
    - "heart problem"
    - "high BP"
    - "sugar disease"
    - "flu symptoms"
    """)

if __name__ == "__main__":
    main()
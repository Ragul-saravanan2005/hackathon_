import streamlit as st
import pandas as pd
import re, os

try:
    from sentence_transformers import SentenceTransformer, util
    from rapidfuzz import fuzz
    ML_AVAILABLE = True
except ImportError as e:
    st.warning(f"ML libraries not available: {e}")
    ML_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🚀 SurveyX - Survey Data Management",
    page_icon="🚀",
    layout="wide"
)

# ----------------------------
# Load datasets
# ----------------------------
@st.cache_data
def load_nco_data():
    try:
        return pd.read_csv("data/MOCK_DATA_with_NCO.csv")
    except FileNotFoundError:
        st.error("NCO data file not found. Please ensure MOCK_DATA_with_NCO.csv exists in the data folder.")
        return pd.DataFrame()

@st.cache_data
def load_survey_data():
    try:
        ctl_file_path = "data/survey_data.ctl"
        with open(ctl_file_path, "r", encoding="utf-8") as f:
            ctl_content = f.read()

        infile_match = re.search(r"INFILE\s+'([^']+)'", ctl_content, re.IGNORECASE)
        if not infile_match:
            return pd.DataFrame()

        raw_path = infile_match.group(1).strip()
        if not os.path.isabs(raw_path):
            data_file_path = os.path.join(os.path.dirname(ctl_file_path), raw_path)
        else:
            data_file_path = raw_path

        if not os.path.exists(data_file_path):
            return pd.DataFrame()

        delimiter_match = re.search(r"FIELDS TERMINATED BY\s+'([^']+)'", ctl_content, re.IGNORECASE)
        delimiter = delimiter_match.group(1) if delimiter_match else ","

        try:
            df = pd.read_csv(data_file_path, delimiter=delimiter, encoding="utf-8", quotechar='"')
            return df
        except UnicodeDecodeError:
            return pd.read_csv(data_file_path, delimiter=delimiter, encoding="latin-1", quotechar='"')
    except Exception as e:
        st.error(f"Error loading survey data: {e}")
        return pd.DataFrame()

# ----------------------------
# Load model + embeddings (with caching)
# ----------------------------
@st.cache_resource
def load_model():
    if ML_AVAILABLE:
        try:
            with st.spinner("Loading SentenceTransformer model..."):
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                st.success("Model loaded successfully!")
                return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        return None

def search_occupation(query, top_k, nco_df, model):
    if not ML_AVAILABLE or model is None:
        # Fallback to simple text matching
        temp_df = nco_df.copy()
        temp_df['simple_score'] = temp_df['occupation_title'].apply(
            lambda title: 1.0 if query.lower() in title.lower() else 0.0
        )
        results = temp_df[temp_df['simple_score'] > 0].sort_values(by='simple_score', ascending=False).head(top_k)
        if results.empty:
            return pd.DataFrame({'occupation_title': ['No matches found'], 'nco_code': ['N/A'], 'final_score': [0.0]})
        return results[['occupation_title', 'nco_code', 'simple_score']].rename(columns={'simple_score': 'final_score'})
    
    try:
        query_emb = model.encode(query, convert_to_tensor=True)
        temp_df = nco_df.copy()
        
        # Create embeddings if they don't exist
        if 'embeddings' not in temp_df.columns:
            with st.spinner("Creating embeddings..."):
                temp_df['embeddings'] = temp_df['occupation_title'].apply(
                    lambda x: model.encode(str(x), convert_to_tensor=True)
                )
        
        temp_df['semantic_score'] = temp_df['embeddings'].apply(
            lambda emb: float(util.cos_sim(query_emb, emb))
        )
        temp_df['fuzzy_score'] = temp_df['occupation_title'].apply(
            lambda title: fuzz.token_sort_ratio(query.lower(), title.lower()) / 100
        )
        temp_df['final_score'] = 0.7 * temp_df['semantic_score'] + 0.3 * temp_df['fuzzy_score']
        results = temp_df.sort_values(by='final_score', ascending=False).head(top_k)
        return results[['occupation_title', 'nco_code', 'final_score']]
    except Exception as e:
        st.error(f"Error in search_occupation: {e}")
        return pd.DataFrame({'occupation_title': ['Search error'], 'nco_code': ['N/A'], 'final_score': [0.0]})

# ----------------------------
# Main App
# ----------------------------
def main():
    st.title("🚀 SurveyX - Survey Data Management System")
    st.markdown("### Comprehensive survey data management with AI-powered occupation search")

    # Load data
    nco_df = load_nco_data()
    survey_df = load_survey_data()
    model = load_model()

    # Sidebar
    st.sidebar.title("Navigation")
    tab_option = st.sidebar.selectbox(
        "Choose a section:",
        ["📊 Survey Data Explorer", "🔍 NCO Occupation Search", "ℹ️ About"]
    )

    if tab_option == "📊 Survey Data Explorer":
        st.header("📊 Survey Data Explorer")
        
        if survey_df.empty:
            st.warning("⚠️ Survey data not loaded.")
        else:
            st.success(f"✅ Survey data loaded successfully! Total records: {len(survey_df)}")
            
            # Preview Data
            st.subheader("Data Preview")
            if st.button("Show Data Preview"):
                st.dataframe(survey_df.head(10), use_container_width=True)
                st.info(f"Showing first 10 records out of {len(survey_df)} total records")

            # Search Data
            st.subheader("Search Survey Data")
            col1, col2 = st.columns(2)
            
            with col1:
                search_column = st.selectbox("Select column to search:", survey_df.columns.tolist())
            with col2:
                search_value = st.text_input("Enter search value:")
            
            if st.button("Search") and search_value.strip():
                results = survey_df[survey_df[search_column].astype(str).str.contains(search_value, case=False, na=False)]
                st.success(f"Found {len(results)} records")
                if not results.empty:
                    st.dataframe(results, use_container_width=True)
                else:
                    st.info("No matching records found.")

    elif tab_option == "🔍 NCO Occupation Search":
        st.header("🔍 NCO Occupation Search")
        st.markdown("### Multilingual Job Title Matching")
        
        if nco_df.empty:
            st.warning("⚠️ NCO data not loaded.")
        else:
            st.success(f"✅ NCO data loaded successfully! {len(nco_df)} occupations available")
            
            # Search Interface
            col1, col2 = st.columns([3, 1])
            with col1:
                job_query = st.text_input(
                    "Enter job title:", 
                    placeholder="e.g. Software Engineer / மென்பொருள் பொறியாளர்"
                )
            with col2:
                top_k = st.slider("Number of results:", 1, 10, 3)
            
            if st.button("Search Occupations") and job_query.strip():
                with st.spinner("Searching..."):
                    results = search_occupation(job_query, top_k, nco_df, model)
                
                if not results.empty and results.iloc[0]['occupation_title'] != 'No matches found':
                    st.success(f"Found {len(results)} matches")
                    # Format the results nicely
                    results_display = results.copy()
                    results_display['final_score'] = results_display['final_score'].round(3)
                    st.dataframe(results_display, use_container_width=True)
                    
                    # Show additional info
                    if ML_AVAILABLE and model is not None:
                        st.info("🤖 Results generated using AI-powered semantic search + fuzzy matching")
                    else:
                        st.info("🔍 Results generated using simple text matching (fallback mode)")
                else:
                    st.warning("No matching occupations found. Try different keywords.")

    elif tab_option == "ℹ️ About":
        st.header("ℹ️ About SurveyX")
        
        st.markdown("""
        **SurveyX** is a comprehensive survey data management system with the following features:
        
        ### 🌟 Key Features
        - **📊 Data Exploration**: Browse and search through survey datasets
        - **🔍 AI-Powered Search**: Multilingual occupation matching using advanced ML models
        - **🗄️ Database Integration**: Support for Oracle database connectivity
        - **📈 Analytics**: Advanced filtering and search capabilities
        - **🌐 Web Interface**: User-friendly interface for data interaction
        
        ### 🛠️ Technology Stack
        - **Frontend**: Streamlit
        - **AI/ML**: Sentence Transformers, RapidFuzz
        - **Data Processing**: Pandas
        - **Database**: Oracle Database support
        
        ### 📊 Data Information
        """)
        
        if not survey_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Survey Records", len(survey_df))
            with col2:
                st.metric("Survey Data Columns", len(survey_df.columns))
                
        if not nco_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("NCO Occupations", len(nco_df))
            with col2:
                st.metric("ML Model Status", "✅ Active" if ML_AVAILABLE and model else "⚠️ Fallback")

        st.markdown("""
        ### 🚀 Deployment
        This application is deployed on Streamlit Share for easy access and collaboration.
        
        ### 📝 Usage
        1. **Survey Data Explorer**: Browse and search your survey data
        2. **NCO Occupation Search**: Find matching occupations using AI-powered search
        3. **Data Analytics**: Explore patterns and insights in your data
        """)

if __name__ == "__main__":
    main()

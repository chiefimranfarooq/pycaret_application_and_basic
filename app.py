import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AutoML with PyCaret",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'final_model' not in st.session_state:
    st.session_state.final_model = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'comparison_df' not in st.session_state:
    st.session_state.comparison_df = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def detect_problem_type(df, target_col):
    """
    Automatically detect if the problem is classification or regression
    based on target variable characteristics
    """
    target_series = df[target_col]
    unique_values = target_series.nunique()
    total_values = len(target_series)
    unique_ratio = unique_values / total_values
    
    # Check data type
    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    is_categorical = pd.api.types.is_categorical_dtype(target_series) or pd.api.types.is_object_dtype(target_series)
    
    # Decision logic
    if is_categorical:
        return "classification"
    elif is_numeric:
        # If unique values are less than 10 and ratio is low, likely classification
        if unique_values <= 10 and unique_ratio < 0.05:
            return "classification"
        # If continuous values, likely regression
        else:
            return "regression"
    else:
        return "classification"  # Default to classification for other types

def run_classification_pipeline(df, target_col):
    """Run PyCaret classification pipeline"""
    from pycaret.classification import setup, compare_models, finalize_model, pull
    
    with st.spinner('🔄 Setting up classification environment...'):
        # Initialize setup
        clf = setup(
            df, 
            target=target_col,
            session_id=123,
            verbose=False
        )
        st.success('✅ Setup completed successfully!')
    
    with st.spinner('🔄 Comparing models... This may take a few minutes...'):
        # Compare models
        best_model = compare_models(n_select=1)
        comparison_df = pull()
        st.success('✅ Model comparison completed!')
    
    with st.spinner('🔄 Finalizing best model...'):
        # Finalize model
        final_model = finalize_model(best_model)
        st.success('✅ Model finalized!')
    
    return final_model, comparison_df, best_model

def run_regression_pipeline(df, target_col):
    """Run PyCaret regression pipeline"""
    from pycaret.regression import setup, compare_models, finalize_model, pull
    
    with st.spinner('🔄 Setting up regression environment...'):
        # Initialize setup
        reg = setup(
            df, 
            target=target_col,
            session_id=123,
            verbose=False
        )
        st.success('✅ Setup completed successfully!')
    
    with st.spinner('🔄 Comparing models... This may take a few minutes...'):
        # Compare models
        best_model = compare_models(n_select=1)
        comparison_df = pull()
        st.success('✅ Model comparison completed!')
    
    with st.spinner('🔄 Finalizing best model...'):
        # Finalize model
        final_model = finalize_model(best_model)
        st.success('✅ Model finalized!')
    
    return final_model, comparison_df, best_model

def main():
    # Header
    st.markdown("<h1>🤖 AutoML with PyCaret</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; padding-bottom: 2rem;'>
            <p style='font-size: 1.2rem; color: #555;'>
                Upload your dataset and let AI find the best machine learning model automatically!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/pycaret/pycaret/master/docs/images/logo.png", width=200)
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("""
            This application uses **PyCaret** to automatically:
            - Detect problem type (Classification/Regression)
            - Compare multiple ML models
            - Select the best performing model
            - Provide detailed reports
        """)
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
            1. Upload your CSV file
            2. Select the target variable
            3. Click 'Run Analysis'
            4. View results and best model
        """)
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
            - Ensure your data is clean
            - Remove unnecessary columns
            - Handle missing values beforehand
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📤 Upload Dataset")
    
    with col2:
        if st.session_state.df is not None:
            if st.button("🗑️ Clear Data", use_container_width=True):
                st.session_state.df = None
                st.session_state.model_trained = False
                st.session_state.final_model = None
                st.session_state.problem_type = None
                st.session_state.comparison_df = None
                st.session_state.best_model = None
                st.rerun()
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with your dataset"
    )
    
    # Handle file upload
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.model_trained = False  # Reset model state when new data is loaded
    
    if st.session_state.df is not None:
        try:
            df = st.session_state.df
            
            # Display dataset info
            st.markdown("## 📊 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📏 Rows", df.shape[0])
            with col2:
                st.metric("📐 Columns", df.shape[1])
            with col3:
                st.metric("🔢 Numeric", df.select_dtypes(include=[np.number]).shape[1])
            with col4:
                st.metric("📝 Categorical", df.select_dtypes(include=['object']).shape[1])
            
            # Show data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Show data types
            with st.expander("🔍 View Column Details"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values,
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
            
            # Target variable selection
            st.markdown("## 🎯 Configure Machine Learning Task")
            target_col = st.selectbox(
                "Select Target Variable",
                options=df.columns.tolist(),
                help="Choose the column you want to predict"
            )
            
            if target_col:
                # Detect problem type
                problem_type = detect_problem_type(df, target_col)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                        <div class='info-box'>
                            <h3 style='margin:0; color: #0c5460;'>🔍 Detected Problem Type</h3>
                            <p style='font-size: 1.5rem; margin: 0.5rem 0; font-weight: bold;'>{problem_type.upper()}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class='info-box'>
                            <h3 style='margin:0; color: #0c5460;'>📊 Target Statistics</h3>
                            <p style='margin: 0.5rem 0;'>Unique Values: <strong>{df[target_col].nunique()}</strong></p>
                            <p style='margin: 0;'>Missing Values: <strong>{df[target_col].isnull().sum()}</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Manual override option
                override = st.checkbox("Override automatic detection", value=False)
                if override:
                    problem_type = st.radio(
                        "Select Problem Type",
                        options=["classification", "regression"],
                        index=0 if problem_type == "classification" else 1
                    )
                
                # Run analysis button
                st.markdown("---")
                if st.button("🚀 Run Analysis", use_container_width=True):
                    st.markdown("## 🔬 Analysis Results")
                    
                    try:
                        if problem_type == "classification":
                            st.info("🎯 Running Classification Pipeline...")
                            final_model, comparison_df, best_model = run_classification_pipeline(df, target_col)
                            
                            # Store results in session state
                            st.session_state.final_model = final_model
                            st.session_state.comparison_df = comparison_df
                            st.session_state.best_model = best_model
                            st.session_state.problem_type = problem_type
                            st.session_state.model_trained = True
                            
                        else:  # regression
                            st.info("📈 Running Regression Pipeline...")
                            final_model, comparison_df, best_model = run_regression_pipeline(df, target_col)
                            
                            # Store results in session state
                            st.session_state.final_model = final_model
                            st.session_state.comparison_df = comparison_df
                            st.session_state.best_model = best_model
                            st.session_state.problem_type = problem_type
                            st.session_state.model_trained = True
                        
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")
                        st.info("💡 Try cleaning your data or selecting a different target variable.")
                
                # Display results if model is trained
                if st.session_state.model_trained:
                    st.markdown("---")
                    st.markdown("## 🔬 Analysis Results")
                    
                    # Display results
                    st.markdown("### 🏆 Model Comparison Results")
                    st.dataframe(st.session_state.comparison_df, use_container_width=True)
                    
                    st.markdown("### 🥇 Best Model")
                    st.markdown(f"""
                        <div class='success-box'>
                            <h3 style='margin:0;'>Selected Model: {type(st.session_state.best_model).__name__}</h3>
                            <p>This model performed best on your {st.session_state.problem_type} task.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Model details
                    with st.expander("📋 View Model Details"):
                        st.write(st.session_state.final_model)
                    
                    # Save model section
                    st.markdown("---")
                    st.markdown("### 💾 Save Model")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        model_name = st.text_input("Model name (without extension)", value="best_model", key="model_name_input")
                    
                    # Save model using PyCaret
                    if st.button("💾 Save Model to Disk", use_container_width=True, key="save_disk"):
                        try:
                            if st.session_state.problem_type == "classification":
                                from pycaret.classification import save_model
                            else:
                                from pycaret.regression import save_model
                            
                            save_model(st.session_state.final_model, model_name)
                            st.success(f"✅ Model saved as '{model_name}.pkl' in the current directory!")
                        except Exception as e:
                            st.error(f"❌ Error saving model: {str(e)}")
                    
                    # Download model as bytes
                    st.markdown("#### Or download the model directly:")
                    try:
                        # Serialize the model to bytes
                        model_bytes = pickle.dumps(st.session_state.final_model)
                        
                        st.download_button(
                            label="⬇️ Download Model",
                            data=model_bytes,
                            file_name=f"{model_name if 'model_name_input' in st.session_state else 'best_model'}.pkl",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"❌ Error preparing download: {str(e)}")
                    
                    # Instructions for using the model
                    with st.expander("📖 How to use the saved model for predictions"):
                        st.markdown(f"""
                        ### Loading and Using the Model
                        
                        **For {st.session_state.problem_type.title()} tasks:**
                        
                        ```python
                        # Import PyCaret
                        from pycaret.{st.session_state.problem_type} import load_model, predict_model
                        import pandas as pd
                        
                        # Load the saved model
                        loaded_model = load_model('{model_name if 'model_name_input' in st.session_state else 'best_model'}')
                        
                        # Prepare your new data (must have same features as training data)
                        new_data = pd.read_csv('your_new_data.csv')
                        
                        # Make predictions
                        predictions = predict_model(loaded_model, data=new_data)
                        
                        # View predictions
                        print(predictions)
                        ```
                        
                        ### Using with Pickle (Alternative)
                        
                        ```python
                        import pickle
                        import pandas as pd
                        
                        # Load the model
                        with open('{model_name if 'model_name_input' in st.session_state else 'best_model'}.pkl', 'rb') as f:
                            model = pickle.load(f)
                        
                        # Make predictions
                        new_data = pd.read_csv('your_new_data.csv')
                        predictions = model.predict(new_data)
                        print(predictions)
                        ```
                        
                        **Note:** Your new data must have the same features (columns) as the training data, excluding the target variable.
                        """)
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 Please ensure your file is a valid CSV format.")
    
    else:
        # Welcome message when no file is uploaded
        st.markdown("""
            <div style='text-align: center; padding: 3rem; background-color: #f8f9fa; border-radius: 10px; margin: 2rem 0;'>
                <h2 style='color: #555;'>👆 Upload a CSV file to get started</h2>
                <p style='color: #777; font-size: 1.1rem;'>
                    Your data will be automatically analyzed and the best ML model will be found!
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Example datasets
        st.markdown("### 📚 Don't have data? Try these examples:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🌸 Iris Dataset (Classification)", use_container_width=True):
                from pycaret.datasets import get_data
                st.session_state.df = get_data('iris')
                st.session_state.model_trained = False
                st.success("✅ Iris dataset loaded! Scroll up to see the data.")
                st.rerun()
        
        with col2:
            if st.button("💎 Diamond Dataset (Regression)", use_container_width=True):
                from pycaret.datasets import get_data
                st.session_state.df = get_data('diamond')
                st.session_state.model_trained = False
                st.success("✅ Diamond dataset loaded! Scroll up to see the data.")
                st.rerun()
        
        with col3:
            if st.button("🏥 Diabetes Dataset (Classification)", use_container_width=True):
                from pycaret.datasets import get_data
                st.session_state.df = get_data('diabetes')
                st.session_state.model_trained = False
                st.success("✅ Diabetes dataset loaded! Scroll up to see the data.")
                st.rerun()

if __name__ == "__main__":
    main()

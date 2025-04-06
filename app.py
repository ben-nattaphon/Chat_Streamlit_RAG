import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from datetime import datetime
import io
import textwrap
from typing import List, Dict, Any
from streamlit import secrets

def generate_code(question, context, model="gemini-2.0-flash-lite"):
    # ... (โค้ดส่วนต้นคงเดิม) ...
    
    try:
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        
        # ใช้วิธีของคุณ - แทนที่ ``` ด้วย # เพื่อความปลอดภัย
        query = response.text.replace("```", "#")
        return query
        
    except Exception as e:
        return f"Error generating code: {str(e)}"


# Set page configuration
st.set_page_config(
    page_title="Liquor Sales Chatbot",
    page_icon="🍹",
    layout="wide"
)

# Configure Gemini API using the API key from Streamlit secrets
def configure_genai():
    api_key = secrets.get("gemini_api_key", "")
    if not api_key:
        st.error("Google API Key not found in Streamlit secrets. Please add it to your secrets.toml file.")
        st.stop()
    genai.configure(api_key=api_key)

# Load and prepare data
@st.cache_data
def load_data():
    # Load the transactions data
    df = pd.read_csv("transactions.csv")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Convert numeric columns to appropriate types
    df['state_bottle_retail'] = pd.to_numeric(df['state_bottle_retail'])
    df['bottles_sold'] = pd.to_numeric(df['bottles_sold'])
    df['sale_dollars'] = pd.to_numeric(df['sale_dollars'])
    
    return df

@st.cache_data
def load_data_dict():
    # Load the data dictionary
    data_dict = pd.read_csv("data_dict.csv")
    return data_dict

# Function to create context for RAG
def create_context(df, data_dict):
    # Get a sample of data (first 2 rows)
    sample_data = df.head(2).to_string()
    
    # Format data dictionary for context
    data_dict_text = data_dict.to_string()
    
    # Create summary stats
    summary_stats = {
        "total_rows": len(df),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "total_sales": f"${df['sale_dollars'].sum():,.2f}",
        "unique_stores": df['store_name'].nunique(),
        "unique_categories": df['category_name'].nunique(),
        "unique_vendors": df['vendor_name'].nunique(),
        "top_categories": df['category_name'].value_counts().head(5).to_dict(),
        "top_cities": df['city'].value_counts().head(5).to_dict()
    }
    
    summary_text = "\n".join([f"{k}: {v}" for k, v in summary_stats.items()])
    
    return {
        "sample_data": sample_data,
        "data_dict": data_dict_text,
        "summary_stats": summary_text
    }

# Function to generate code from question
def generate_code(question, context, model="gemini-2.0-flash-lite"):
    df_name = "df"
    
    # Create the prompt for the code generation
    prompt = f"""
    You are a helpful Python code generator. Your goal is to write Python code snippets based on the user's question and the provided DataFrame information.
    
    Here's the context:
    
    **User Question:** {question}
    
    **DataFrame Name:** {df_name}
    
    **DataFrame Details:** 
    {context['data_dict']}
    
    **Sample Data (Top 2 Rows):** 
    {context['sample_data']}
    
    **DataFrame Summary:**
    {context['summary_stats']}
    
    **Instructions:**
    1. Write Python code that addresses the user's question by querying or manipulating the DataFrame.
    2. **Crucially, use the **`exec()` function to execute the generated code.
    3. Do not import pandas, it's already imported.
    4. The date column is already converted to datetime.
    5. **Store the result of the executed code in a variable named **`ANSWER`. This variable should hold the answer to the user's question (e.g., a filtered DataFrame, a calculated value, etc.).
    6. Assume the DataFrame is already loaded into a pandas DataFrame object named `{df_name}`. Do not include code to load the DataFrame.
    7. Keep the generated code concise and focused on answering the question.
    8. If the question requires a specific output format (e.g., a list, a single value), ensure the `ANSWER` variable holds that format.
    9. If the result is a DataFrame with many rows, limit it to 10 rows for display purposes.
    
    Return ONLY the Python code without any explanation or markdown formatting.
    """
    
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating code: {str(e)}"

# Function to execute generated code and get results
# Function to execute generated code and get results
def execute_code_and_get_result(code, df):
    # ... (โค้ดส่วนต้นคงเดิม) ...
    
    try:
        # Execute the code
        exec(code, namespace)
        
        # Get the result
        result = namespace.get("ANSWER", "No result was stored in the ANSWER variable")
        
        # Create enhanced explanation
        explain_the_results = f'''
        the user asked {question},
        here is the results {result}
        answer the question and summarize the answer,
        include your opinions of the persona of this customer
        '''
        
        model_instance = genai.GenerativeModel("gemini-2.0-flash-lite")
        explanation_response = model_instance.generate_content(explain_the_results)
        explanation = explanation_response.text
        
        return {"success": True, "result": result, "code": code, "explanation": explanation}
    except Exception as e:
        # ... (โค้ดการจัดการข้อผิดพลาดคงเดิม) ...

# Function to answer question with RAG
def answer_question_with_rag(question, execution_result, context, model="gemini-2.0-flash-lite"):
    if not execution_result["success"]:
        return f"Error executing code: {execution_result['error']}"
    
    # Format the execution result
    if isinstance(execution_result["result"], pd.DataFrame):
        result_str = execution_result["result"].to_string()
    else:
        result_str = str(execution_result["result"])
    
    # Try the new approach with enhanced explanation and summary
    try:
        # First, create a detailed explanation of the results
        explain_the_results = f'''
        The user asked: "{question}"
        
        Here are the results: {result_str}
        
        DataFrame Details: 
        {context['data_dict']}
        
        DataFrame Summary:
        {context['summary_stats']}
        
        Please provide a detailed answer to the user's question based on these results. Your response should:
        1. Directly answer the question
        2. Explain key insights from the data
        3. Provide context about why these results matter
        4. Summarize the findings in a business-friendly way
        5. Add any interesting patterns or anomalies you notice
        6. Include your analysis of what this might tell us about customer preferences or market trends
        
        Make your response conversational and insightful, as if you're a data analyst explaining findings to a colleague.
        '''
        
        # Generate the enhanced explanation
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(explain_the_results)
        
        return response.text.strip()
    except Exception as e:
        # Fallback to a simpler approach if the enhanced explanation fails
        try:
            # Create a more basic RAG prompt
            rag_prompt = f"""
            You are a helpful assistant specialized in analyzing liquor sales data. Answer the following question based on the data execution results provided.
            
            **User Question:** {question}
            
            **Data Execution Results:** 
            {result_str}
            
            Please provide a clear, concise answer to the user's question based on the data results.
            Summarize the key points and insights.
            """
            
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(rag_prompt)
            return response.text.strip()
        except Exception as inner_e:
            return f"Error generating answer: {str(e)}. I tried a simpler approach but encountered: {str(inner_e)}. Here's the raw data:\n{result_str[:500]}..."

# Main app function
def main():
    st.title("🍹 Liquor Sales Data Analysis Chatbot")
    
    # Sidebar for information
    with st.sidebar:
        st.header("About")
        st.write("""
        This chatbot uses Google's Gemini model to analyze liquor sales data.
        You can ask questions about sales trends, popular products, store performance, and more.
        """)
        
        st.divider()
        
        st.header("Sample Questions")
        st.write("""
        ### Try asking:
        - What are the top 5 selling liquor categories by total sales?
        - Which store had the highest sales in February 2025?
        - What's the average number of bottles sold per transaction?
        - Show me sales trends by date for CANADIAN WHISKIES
        - Which city has the most diverse selection of liquor categories?
        """)
    
    # Configure the Gemini API with the API key from secrets
    try:
        configure_genai()
    except Exception as e:
        st.error(f"Error configuring API: {str(e)}")
        return
    
    # Load data
    try:
        with st.spinner("Loading data..."):
            df = load_data()
            data_dict = load_data_dict()
            context = create_context(df, data_dict)
        
        # Display data overview
        with st.expander("Data Overview"):
            st.subheader("Sample Data")
            st.dataframe(df.head(), use_container_width=True)
            
            st.subheader("Data Dictionary")
            st.dataframe(data_dict, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the liquor sales data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                # Generate code
                with st.expander("Generated Code", expanded=False):
                    code = generate_code(prompt, context)
                    st.code(code, language="python")
                
                # Execute code
                execution_result = execute_code_and_get_result(code, df)
                
                if not execution_result["success"]:
                    st.error(f"Error executing code: {execution_result['error']}")
                    response = "I encountered an error while analyzing your data. Please try rephrasing your question."
                else:
                    # Display raw results
                    with st.expander("Raw Execution Result", expanded=False):
                        if isinstance(execution_result["result"], pd.DataFrame):
                            st.dataframe(execution_result["result"], use_container_width=True)
                        else:
                            st.write(execution_result["result"])
                    
                    # Generate RAG answer
                    response = answer_question_with_rag(prompt, execution_result, context)
            
            # Display final response
            response_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

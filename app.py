import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from datetime import datetime
import io
import textwrap
from typing import List, Dict, Any
from streamlit import secrets
import traceback

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

# Original generate_code function (kept as backup)
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
    2. DO NOT use the exec() function or create code string variables. AVOID using triple quotes.
    3. Do not import pandas, it's already imported.
    4. The date column is already converted to datetime.
    5. Store the result in a variable named `ANSWER`. This variable should hold the answer to the user's question (e.g., a filtered DataFrame, a calculated value, etc.).
    6. Assume the DataFrame is already loaded into a pandas DataFrame object named `{df_name}`. Do not include code to load the DataFrame.
    7. Keep the generated code concise and focused on answering the question.
    8. If the question requires a specific output format (e.g., a list, a single value), ensure the `ANSWER` variable holds that format.
    9. If the result is a DataFrame with many rows, limit it to 10 rows for display purposes.
    
    Return ONLY the Python code without any explanation, markdown formatting or code blocks. Your response should start directly with the Python code.
    
    Example of a good response:
    ANSWER = df.groupby('category_name')['sale_dollars'].sum().nlargest(5)
    """
    
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt)
        code = response.text.strip()
        
        # Clean the code to remove markdown formatting if present
        if code.startswith("```python"):
            code = code.split("```python", 1)[1]
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
        
        return code.strip()
    except Exception as e:
        return f"Error generating code: {str(e)}"

# Original execute_code_and_get_result function (kept as backup)
def execute_code_and_get_result(code, df):
    # Create a namespace for execution
    namespace = {"df": df, "pd": pd, "ANSWER": None}
    
    try:
        # Clean the code first
        clean_code = code.strip()
        
        # Remove markdown code blocks if present
        if clean_code.startswith("```python"):
            clean_code = clean_code.split("```python", 1)[1]
        if clean_code.endswith("```"):
            clean_code = clean_code.rsplit("```", 1)[0]
            
        # Remove any code = """ pattern which causes issues
        if 'code = """' in clean_code:
            # Extract the actual code inside the triple quotes
            code_lines = clean_code.split('code = """', 1)[1].split('"""', 1)[0].strip().splitlines()
            # Keep only the lines that should be executed (ignore the ANSWER line if present)
            actual_code = []
            for line in code_lines:
                if line.strip() and not line.strip().startswith('"""'):
                    actual_code.append(line)
            clean_code = "\n".join(actual_code)
        
        # Fix the common exec pattern issue
        if 'exec(code)' in clean_code:
            lines = clean_code.splitlines()
            actual_code = []
            for i, line in enumerate(lines):
                if line.strip().startswith('ANSWER ='):
                    actual_code.append(line)
                elif 'exec(code)' in line:
                    # Skip this line
                    pass
                elif not (line.strip().startswith('code =') or line.strip() == '"""' or line.strip() == "'''"):
                    actual_code.append(line)
            clean_code = "\n".join(actual_code)
        
        # Execute the cleaned code
        exec(clean_code, namespace)
        
        # Get the result
        result = namespace.get("ANSWER", "No result was stored in the ANSWER variable")
        return {"success": True, "result": result, "code": clean_code}
    except Exception as e:
        # For more detailed error debugging
        error_details = traceback.format_exc()
        return {"success": False, "error": str(e), "code": code, "error_details": error_details}

# Original answer_question_with_rag function (kept as backup)
def answer_question_with_rag(question, execution_result, context, model="gemini-2.0-flash-lite"):
    if not execution_result["success"]:
        return f"Error executing code: {execution_result['error']}"
    
    # Format the execution result
    if isinstance(execution_result["result"], pd.DataFrame):
        result_str = execution_result["result"].to_string()
    else:
        result_str = str(execution_result["result"])
    
    # Create the RAG prompt
    rag_prompt = f"""
    You are a helpful assistant specialized in analyzing liquor sales data. Answer the following question based on the data execution results provided.
    
    **User Question:** {question}
    
    **Data Execution Results:** 
    {result_str}
    
    **DataFrame Details:** 
    {context['data_dict']}
    
    **DataFrame Summary:**
    {context['summary_stats']}
    
    Please provide a clear, concise answer to the user's question based on the data results. 
    If the results are unclear or if there's an issue with the data, explain what might be going wrong.
    Use bullet points where appropriate to make the information more readable.
    """
    
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(rag_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Function to generate and execute code, then provide enhanced explanation
def generate_code_and_explain(question, context, df, model="gemini-2.0-flash-lite"):
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
    2. Do not import pandas, it's already imported.
    3. The date column is already converted to datetime.
    4. Store the result in a variable named `ANSWER`. This variable should hold the answer to the user's question.
    5. Assume the DataFrame is already loaded into a pandas DataFrame object named `{df_name}`.
    6. Keep the generated code concise and focused on answering the question.
    7. If the result is a DataFrame with many rows, limit it to 10 rows for display.
    
    Return ONLY the Python code without any explanation.
    """
    
    try:
        # Generate the code
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        
        # Process the response by replacing markdown code blocks with comments
        query = response.text.replace("```", "#")
        
        # Create a namespace for execution
        namespace = {"df": df, "pd": pd, "ANSWER": None}
        
        # Execute the processed code
        try:
            exec(query, namespace)
            
            # Get the result
            result = namespace.get("ANSWER", "No result was stored in the ANSWER variable")
            
            # Generate an enhanced explanation
            explain_the_results = f'''
            The user asked: "{question}"
            
            Here are the results: {str(result)}
            
            Answer the question thoroughly and summarize the key findings.
            Include your analysis of what this might tell us about customer preferences or market trends.
            Provide insights that would be valuable for business decision-making.
            '''
            
            # Get the explanation
            explanation_response = model_instance.generate_content(explain_the_results)
            explanation = explanation_response.text.strip()
            
            return {
                "success": True, 
                "code": query, 
                "result": result,
                "explanation": explanation
            }
            
        except Exception as exec_error:
            import traceback
            error_details = traceback.format_exc()
            return {
                "success": False, 
                "error": str(exec_error), 
                "code": query,
                "error_details": error_details
            }
            
    except Exception as e:
        return {"success": False, "error": f"Error generating code: {str(e)}"}

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
        
        # Option to manually enter API key if there are issues with secrets
        
        
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
        # First try to use secrets
        try:
            configure_genai()
        except Exception as e:
            # If secrets fail, try environment variables
            st.warning(f"Could not configure API with secrets: {str(e)}")
            st.info("Trying environment variables...")
            
            if 'GOOGLE_API_KEY' in os.environ:
                api_key = os.environ.get('GOOGLE_API_KEY')
                st.info("Found API key in environment variables")
                genai.configure(api_key=api_key)
                st.success("API configured using environment variable!")
            else:
                st.error("No API key found in environment variables either.")
                st.info("Please use the API Configuration section in the sidebar to enter your API key manually.")
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
                # Use the combined function for code generation, execution and explanation
                result = generate_code_and_explain(prompt, context, df)
                
                if not result["success"]:
                    st.error(f"Error: {result['error']}")
                    
                    # Show detailed error information for debugging
                    if "error_details" in result:
                        with st.expander("Error Details", expanded=False):
                            st.code(result["error_details"], language="python")
                    
                    # Try a fallback approach
                    st.info("Let me try an alternative approach...")
                    
                    # Use the regular RAG approach as fallback
                    with st.expander("Generated Code", expanded=False):
                        code = generate_code(prompt, context)
                        st.code(code, language="python")
                    
                    execution_result = execute_code_and_get_result(code, df)
                    
                    if execution_result["success"]:
                        st.success("I was able to get a result with an alternative approach!")
                        
                        # Display raw results
                        with st.expander("Raw Execution Result", expanded=False):
                            if isinstance(execution_result["result"], pd.DataFrame):
                                st.dataframe(execution_result["result"], use_container_width=True)
                            else:
                                st.write(execution_result["result"])
                        
                        # Generate RAG answer
                        response = answer_question_with_rag(prompt, execution_result, context)
                    else:
                        response = "I encountered errors while analyzing your data. Please try rephrasing your question."
                else:
                    # The combined approach was successful
                    
                    # Show the generated code
                    with st.expander("Generated Code", expanded=False):
                        st.code(result["code"], language="python")
                    
                    # Display raw results
                    with st.expander("Raw Execution Result", expanded=False):
                        if isinstance(result["result"], pd.DataFrame):
                            st.dataframe(result["result"], use_container_width=True)
                        else:
                            st.write(result["result"])
                    
                    # Use the enhanced explanation
                    response = result["explanation"]
            
            # Display final response
            response_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

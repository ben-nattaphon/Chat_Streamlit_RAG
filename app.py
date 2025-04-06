import pathlib
import textwrap
import google.generativeai as genai
import pandas as pd
###from IPython.display import display
###from IPython.display import Markdown
def to_markdown(text):
  text = text.replace('•', ' *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda
_: True))

import streamlit as st
import google.generativeai as genai

try:
    key = st.secrets['gemini_api_key']
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
 
    if "chat" not in st.session_state:
        st.session_state.chat = model.start_chat(history=[])
    st.title('Gemini Pro Test')
 
    def role_to_streamlit(role: str) -> str:
        if role == 'model':
            return 'assistant'
        else:
            return role
 
    for message in st.session_state.chat.history:
        with st.chat_message(role_to_streamlit(message.role)):
            st.markdown(message.parts[0].text)
 
    if prompt := st.chat_input("Text Here"):
        st.chat_message('user').markdown(prompt)
        response = st.session_state.chat.send_message(prompt)
        with st.chat_message('assistant'):
            st.markdown(response.text)
except Exception as e:
    st.error(f'An error occurred {e}')


model = genai.GenerativeModel('gemini-2.0-flash-lite')

# Commented out IPython magic to ensure Python compatibility.
# %%time
# response = model.generate_content("Who is Mild Charinrat")
# to_markdown(response.text)

transaction_df = pd.read_csv('/content/transactions.csv')
example_record = transaction_df.head(2).to_string()
data_dict_df = pd.read_csv('/content/data_dict.csv')
data_dict_text = '\n'.join('- '+ data_dict_df['column_name']+
                  ': '+data_dict_df['data_type']+
                  '. '+data_dict_df['description'])
df_name = 'transaction_df'

print(data_dict_text)

question = 'How many total sale in jan 2025?'

prompt = f"""
You are a helpful Python code generator.
Your goal is to write Python code snippets based on the user's question
and the provided DataFrame information.
Here's the context:
**User Question:**
{question}
**DataFrame Name:**
{df_name}
**DataFrame Details:**
{data_dict_text}
**Sample Data (Top 2 Rows):**
{example_record}

**Instructions:**
1. Write Python code that addresses the user's question by querying or
manipulating the DataFrame.
2. **Crucially, use the `exec()` function to execute the generated
code.**
3. Do not import pandas
4. Change date column type to datetime
5. **Store the result of the executed code in a variable named
`ANSWER`.**
This variable should hold the answer to the user's question (e.g.,
a filtered DataFrame, a calculated value, etc.).
6. Assume the DataFrame is already loaded into a pandas DataFrame object
named `{df_name}`. Do not include code to load the DataFrame.
7. Keep the generated code concise and focused on answering the question.
8. If the question requires a specific output format (e.g., a list, a
single value), ensure the `query_result` variable holds that format.

**Example:**
If the user asks: "Show me the rows where the 'age' column is
greater than 30."
And the DataFrame has an 'age' column.
The generated code should look something like this (inside the
`exec()` string):
```python
query_result = {df_name}[{df_name}['age'] > 30]
"""

response = model.generate_content(prompt)

to_markdown(response.text)

query = response.text.replace("```", "#")
exec(query)
explain_the_results = f'''
the user asked {question},
here is the results {ANSWER}
answer the question and summarize the answer,
include your opinions of the persona of this customer
'''
response = model.generate_content(explain_the_results)

response = model.generate_content(explain_the_results)

to_markdown(response.text)

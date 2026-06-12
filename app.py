import streamlit as st
from Financial_RAG import ask_financial_assistant

st.set_page_config(
    page_title="Financial RAG Assistant",
    layout="wide"
)

st.title("Financial RAG Assistant")

query = st.text_input(
    "Ask a financial question"
)

if st.button("Analyze"):

    result = ask_financial_assistant(query)

    st.write(result["answer"])

    st.subheader("Sources")

    for source in result["sources"]:
        st.write(source)

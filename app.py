
import streamlit as st

from rag_pipeline import ask_question


# Page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚"
)

# App title
st.title("📚 PDF RAG Chatbot")

st.write("Ask questions about your uploaded PDF.")


# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
question = st.chat_input("Ask a question about your PDF...")


# Process question
if question:

    # Display user question
    st.chat_message("user").markdown(question)

    # Save user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching PDF and generating answer..."):
            answer = ask_question(question)

        st.markdown(answer)

    # Save assistant answer
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

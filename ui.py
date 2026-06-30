import streamlit as st
import requests

st.title("🔐 Compliance AI Assistant")

question = st.text_input("Ask a question")

if st.button("Get Answer"):
    response = requests.post(
        "http://127.0.0.1:8000/ask",
        json={"question": question}
    )

    data = response.json()

    st.subheader("Answer")
    st.write(data["answer"])

    st.subheader("Sources")
    st.write(data["sources"])
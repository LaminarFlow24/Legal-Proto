import re
import streamlit as st

from utils.document_util import extract_from_pdf, extract_from_scanned_pdf, tag_documents
from llm.llm_chain import LLM
from vectorstore.faiss_store import FAISSStore


def main():
    st.set_page_config(page_title="ClauseSense", layout="wide", page_icon="♦")
    st.markdown(
        "<h1 style='text-align: center; font-weight:bold; font-family:comic sans ms; padding-top: 0rem;'> \
            Legal Document Summarization</h1>",
        unsafe_allow_html=True,
    )

    clauses = {}
    clause_count = st.sidebar.slider(label="Number of clause", min_value=1, max_value=5)

    ## Upload document
    file = st.file_uploader(label="Upload your document (pdf)", type="pdf")
    if file:
        button = st.button("Submit")

        llm = LLM()
        llm_chain = llm.get_chain()

        st.subheader("Enter clauses that you want to summarize 👇")
        cols = st.columns(clause_count)
        for i, x in enumerate(cols):
            clauses[x.text_input(label=f"clause {i+1}", value="", key=i)] = []

    if file and button:
        st.divider()
        with st.spinner("Generating summaries..."):
            data = extract_from_scanned_pdf(file)
            tagged_documents = tag_documents(data)

            faiss_store = FAISSStore()
            faiss_store.add_documents(tagged_documents)

            for k, v in clauses.items():
                relevant_documents = faiss_store.search(query=k)
                print(relevant_documents)
                v.append(
                    llm.summarize(
                        relevant_documents=relevant_documents,
                        query=k,
                        chain=llm_chain,
                        class_name="Not_Applicable",  # FAISSStore doesn't create classes
                    )
                )

        # expander view
        items = list(clauses.items())
        for i, x in enumerate(cols):
            x.subheader(items[i][0])
            text = "".join(items[i][1][0])
            x.text_area("Summary", text, key=i+10, height=400)


if __name__ == "__main__":
    main()
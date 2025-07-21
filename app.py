import streamlit as st

# Assuming these imports are in your project structure
from utils.document_util import extract_from_scanned_pdf, tag_documents
from llm.llm_chain import LLM
from vectorstore.faiss_store import FAISSStore

def main():
    # --- 1. PAGE CONFIGURATION ---
    # Use a more professional title, an appropriate icon, and a centered layout.
    st.set_page_config(
        page_title="ClauseWise Summarizer",
        page_icon="⚖️",
        layout="centered"
    )

    # --- 2. HEADER AND DESCRIPTION ---
    # Use Streamlit's native elements for a cleaner look. Avoid raw HTML/CSS.
    st.title("ClauseWise Summarizer ⚖️")
    st.markdown(
        "Welcome! This tool helps you quickly summarize specific clauses from legal documents. "
        "Just upload your PDF, enter the clauses you're interested in, and get AI-powered summaries."
    )
    st.divider()

    # --- 3. SIDEBAR FOR CONTROLS ---
    # Grouping controls in the sidebar keeps the main interface clean.
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # File uploader in the sidebar
        uploaded_file = st.file_uploader(
            label="Upload your document",
            type="pdf",
            help="Upload a scanned or digital PDF document."
        )
        
        # Slider to select the number of clauses
        clause_count = st.slider(
            label="How many clauses do you want to summarize?",
            min_value=1,
            max_value=10,
            value=2 # A sensible default
        )

    # --- 4. MAIN CONTENT AREA ---
    if not uploaded_file:
        st.info("Please upload a PDF document using the sidebar to get started.")
        st.stop()

    # Use a form to batch inputs and prevent re-running on every keystroke.
    with st.form("clause_form"):
        st.subheader("Enter the Clauses to Summarize")
        st.markdown("Type or paste the exact clause text you want to find and summarize from the document.")

        # Dynamically create text input fields based on the slider
        clauses_to_find = []
        for i in range(clause_count):
            clauses_to_find.append(st.text_input(
                label=f"Clause {i+1}",
                key=f"clause_input_{i}"
            ))

        # The submit button for the form
        submitted = st.form_submit_button("✨ Generate Summaries")

    # --- 5. PROCESSING AND DISPLAYING RESULTS ---
    # This block runs only when the form is submitted.
    if submitted:
        # Check if all clause inputs are non-empty
        if not all(clauses_to_find):
            st.error("Please fill in all the clause fields before submitting.")
            st.stop()

        with st.spinner("Analyzing document and generating summaries... Please wait."):
            try:
                # Initialize LLM and Vector Store
                llm = LLM()
                llm_chain = llm.get_chain()
                faiss_store = FAISSStore()

                # Process the document
                data = extract_from_scanned_pdf(uploaded_file)
                tagged_documents = tag_documents(data)
                faiss_store.add_documents(tagged_documents)

                # Store summaries in a dictionary
                summaries = {}
                for query in clauses_to_find:
                    if query: # Ensure query is not an empty string
                        relevant_documents = faiss_store.search(query=query)
                        summary = llm.summarize(
                            relevant_documents=relevant_documents,
                            query=query,
                            chain=llm_chain,
                            class_name="Not_Applicable", # As per original logic
                        )
                        summaries[query] = "".join(summary)

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.stop()

        # Display results in a clean, expandable format
        st.divider()
        st.subheader("✅ Summaries")
        if not summaries:
            st.warning("Could not generate any summaries. Try rephrasing your clauses or checking the document.")
        else:
            for query, summary_text in summaries.items():
                with st.expander(f"**Summary for:** `{query[:80]}...`"):
                    st.write(summary_text)


if __name__ == "__main__":
    main()

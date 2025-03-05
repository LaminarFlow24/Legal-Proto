from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class FAISSStore:
    def __init__(self):
        # Use HuggingFaceEmbeddings instead of OpenAIEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.documents = []

    def add_documents(self, tagged_documents: list):
        # tagged_documents is a list of dicts with a "heading" and "documents" key.
        # Flatten documents from all tags into one list.
        docs = []
        for item in tagged_documents:
            docs.extend(item["documents"])
        self.documents.extend(docs)
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        else:
            self.vectorstore.add_texts([doc.page_content for doc in docs])

    def search(self, query: str, k: int = 4) -> list:
        if self.vectorstore is None:
            return []
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
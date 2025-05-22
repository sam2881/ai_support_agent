# agent/rag/vector_store.py
import os
import pickle
import logging
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Configure logging for this module
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_dir: str = "agent/rag"):
        """
        Initializes the VectorStore, attempting to load an existing FAISS index
        and associated documents. If not found, it prepares for creation.

        Args:
            index_dir (str): The directory where the FAISS index and documents are stored.
        """
        self.index_dir = index_dir
        self.docs_path = os.path.join(index_dir, "docs.pkl")
        
        # Initialize OpenAIEmbeddings as the embedding model
        # Ensure OPENAI_API_KEY is set in your environment or through settings.
        self.embedding_model = OpenAIEmbeddings() 
        
        self.faiss_store: FAISS = None
        self.docs: List[Document] = [] # Stores original Langchain Document objects

        index_file = os.path.join(index_dir, "index.faiss")

        if os.path.exists(index_file) and os.path.exists(self.docs_path):
            try:
                logger.info("📦 Loading FAISS index and documents from disk...")
                # Load FAISS index
                self.faiss_store = FAISS.load_local(
                    folder_path=index_dir,
                    embeddings=self.embedding_model,
                    allow_dangerous_deserialization=True # Necessary for loading from disk
                )
                
                # Load associated documents
                with open(self.docs_path, "rb") as f:
                    self.docs = pickle.load(f)
                
                logger.info(f"✅ Loaded FAISS index with {len(self.docs)} documents.")
            except Exception as e:
                logger.error(f"❌ Error loading FAISS index or documents: {e}. Starting fresh.", exc_info=True)
                self.faiss_store = None
                self.docs = []
        else:
            logger.info("🆕 No existing FAISS index or docs.pkl found. A new one will be created on first add_document().")

    def add_document(self, doc_data: Dict[str, Any]):
        """
        Adds a single document to the vector store.
        It expects a dictionary containing 'page_content' and 'metadata'.
        The index is updated and saved locally after each addition.

        Args:
            doc_data (Dict[str, Any]): A dictionary representing the document.
                                      Expected keys: 'page_content' (str) and 'metadata' (Dict).
        """
        if not isinstance(doc_data, dict) or "page_content" not in doc_data or "metadata" not in doc_data:
            logger.error("❌ Invalid document format. Must be a dictionary with 'page_content' and 'metadata'.")
            return

        doc = Document(page_content=doc_data["page_content"], metadata=doc_data["metadata"])
        self.docs.append(doc)

        os.makedirs(self.index_dir, exist_ok=True)

        if self.faiss_store is None:
            logger.info("🚀 Creating new FAISS index from scratch...")
            self.faiss_store = FAISS.from_documents([doc], self.embedding_model)
        else:
            logger.info(f"Adding document to existing FAISS index: '{doc.page_content[:50]}...'")
            self.faiss_store.add_documents([doc])

        # Save updated FAISS index
        try:
            self.faiss_store.save_local(self.index_dir)
            # Save updated doc list
            with open(self.docs_path, "wb") as f:
                pickle.dump(self.docs, f)
            logger.info(f"✅ Document added and FAISS index saved locally. Total documents: {len(self.docs)}")
        except Exception as e:
            logger.error(f"❌ Failed to save FAISS index or documents: {e}", exc_info=True)

    def query(self, text: str, k: int = 3) -> List[Document]:
        """
        Performs a similarity search against the FAISS index to find relevant documents.

        Args:
            text (str): The query text to search for.
            k (int): The number of top similar documents to retrieve.

        Returns:
            List[Document]: A list of `langchain.schema.Document` objects that are
                            most similar to the query text. Returns an empty list
                            if the FAISS index is not initialized.
        """
        if not self.faiss_store:
            logger.warning("⚠️ FAISS index is not initialized or empty. Cannot perform search.")
            return []

        try:
            results = self.faiss_store.similarity_search(text, k=k)
            logger.info(f"🔎 Queried FAISS for '{text[:50]}...'. Found {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"❌ Error during FAISS similarity search: {e}", exc_info=True)
            return []

    # The query_with_graph method seems to imply integration with GraphClient,
    # but the current design passes enrichment from MainAgent.
    # If this method is intended to be called standalone and needs a GraphClient,
    # you'd need to instantiate it here or pass it in init.
    # For now, I'm commenting it out as it's not used in the `main_agent.py` logic provided,
    # and its inclusion here might cause a circular dependency or uninitialized attribute error.
    # def query_with_graph(self, query_text: str, k: int = 3):
    #     """
    #     Returns:
    #         - Top K FAISS documents
    #         - Graph context from Neo4j
    #     """
    #     if not self.faiss_store:
    #         raise ValueError("FAISS index is empty. Cannot perform search.")

    #     top_docs = self.faiss_store.similarity_search(query_text, k=k)

    #     # Enrich with graph context (optional — only if `self.graph` is defined)
    #     # This assumes GraphClient is available and initialized.
    #     # If you want this, you might need to pass a GraphClient instance to VectorStore
    #     # or handle the import and instantiation here.
    #     graph_context = self.graph.query_similar_issues(query_text) if hasattr(self, 'graph') else None
    #     return top_docs, graph_context


# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Configure basic logging for the example run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting VectorStore example...")

    # Ensure the 'agent/rag' directory exists for saving
    os.makedirs("agent/rag", exist_ok=True)

    # Initialize the VectorStore
    vs = VectorStore(index_dir="agent/rag")

    # Add some dummy documents
    print("\n--- Adding documents ---")
    doc1 = {
        "page_content": "Airflow DAG `my_data_pipeline` failed due to a `ZeroDivisionError` in task `transform_data`. This is a common data quality issue.",
        "metadata": {"category": "airflow", "root_cause": "data_quality", "solution": "add input validation", "title": "ZeroDivisionError"}
    }
    vs.add_document(doc1)

    doc2 = {
        "page_content": "Permission denied when attempting to write to S3 bucket `my-bucket-logs`. Check IAM roles and bucket policies.",
        "metadata": {"category": "access", "root_cause": "iam_permissions", "solution": "update IAM policy", "title": "S3 Access Denied"}
    }
    vs.add_document(doc2)

    doc3 = {
        "page_content": "Kubernetes pod for Airflow worker `airflow-worker-xyz` is stuck in `CrashLoopBackOff`. Usually indicates a container startup error or OOMKilled.",
        "metadata": {"category": "devops", "root_cause": "kubernetes_pod_failure", "solution": "check pod logs, increase resources", "title": "K8s Pod CrashLoop"}
    }
    vs.add_document(doc3)

    doc4 = {
        "page_content": "Airflow scheduler is not running. No new DAG runs are being triggered. Check scheduler logs for errors.",
        "metadata": {"category": "devops", "root_cause": "airflow_component_down", "solution": "restart airflow scheduler", "title": "Airflow Scheduler Down"}
    }
    vs.add_document(doc4)
    
    # Test queries
    print("\n--- Performing queries ---")
    query_text_1 = "Airflow task failed with a division by zero."
    results_1 = vs.query(query_text_1, k=2)
    print(f"\nQuery: '{query_text_1}'")
    if results_1:
        for i, doc in enumerate(results_1):
            print(f"  Result {i+1}: Content: '{doc.page_content[:70]}...', Category: {doc.metadata.get('category')}")
    else:
        print("  No results found.")

    query_text_2 = "Can't access S3 from my Airflow DAG."
    results_2 = vs.query(query_text_2, k=1)
    print(f"\nQuery: '{query_text_2}'")
    if results_2:
        for i, doc in enumerate(results_2):
            print(f"  Result {i+1}: Content: '{doc.page_content[:70]}...', Category: {doc.metadata.get('category')}")
    else:
        print("  No results found.")

    query_text_3 = "Kubernetes pods are not starting up."
    results_3 = vs.query(query_text_3, k=2)
    print(f"\nQuery: '{query_text_3}'")
    if results_3:
        for i, doc in enumerate(results_3):
            print(f"  Result {i+1}: Content: '{doc.page_content[:70]}...', Category: {doc.metadata.get('category')}")
    else:
        print("  No results found.")

    # Clean up created files (optional)
    # print("\n--- Cleaning up created index files ---")
    # if os.path.exists("agent/rag/index.faiss"):
    #     os.remove("agent/rag/index.faiss")
    # if os.path.exists("agent/rag/index.pkl"): # Langchain also creates a .pkl alongside .faiss
    #     os.remove("agent/rag/index.pkl")
    # if os.path.exists("agent/rag/docs.pkl"):
    #     os.remove("agent/rag/docs.pkl")
    # if os.path.exists("agent/rag"): # Only remove directory if empty
    #     try:
    #         os.rmdir("agent/rag")
    #     except OSError:
    #         pass # Directory not empty

    logger.info("VectorStore example finished.")
# scripts/build_faiss_index.py

import os
import csv
import logging
from dotenv import load_dotenv

# Ensure the project root is in sys.path for module imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agent.rag.vector_store import VectorStore, Document # Import Document and VectorStore
from agent.workflows.config import settings # Assuming settings will contain CSV_PATH

load_dotenv()

logger = logging.getLogger(__name__)

# Configure logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Assuming CSV_PATH is defined in settings or as a constant here
# If it's in settings, uncomment the line below and ensure settings is properly configured.
# CSV_PATH = settings.KNOWLEDGE_BASE_CSV_PATH 
CSV_PATH = "ingestion/errors.csv" # Defaulting for example, adjust as needed

def load_knowledge_base_from_csv(csv_path: str) -> list[Document]:
    """
    Loads error and remediation documents from a CSV file and converts them
    into a list of Document objects suitable for the VectorStore.

    The CSV is expected to have 'error', 'remediation', and 'category' columns.
    """
    documents = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Ensure required columns exist
                if 'error' not in row or 'remediation' not in row or 'category' not in row:
                    logger.warning(f"Skipping row {i+1} due to missing required columns (error, remediation, category): {row}")
                    continue
                
                error_text = row['error']
                remediation_text = row['remediation']
                category = row['category']
                
                # Create page_content for embedding. Combine error and remediation.
                # The LLM will use this combined text for similarity search.
                page_content = f"Error: {error_text}\nRemediation: {remediation_text}"

                # Metadata for filtering and retrieval
                metadata = {
                    'error_summary': error_text,
                    'solution_description': remediation_text,
                    'category': category,
                    'source': 'static_knowledge_base_csv',
                    'labels': [category, "knowledge-base", "auto-labeled"] # Structured labels
                }
                documents.append(Document(page_content=page_content, metadata=metadata))
        logger.info(f"Successfully loaded {len(documents)} documents from {csv_path}")
    except FileNotFoundError:
        logger.error(f"Error: CSV file not found at {csv_path}. Please ensure the path is correct.", exc_info=True)
    except Exception as e:
        logger.error(f"An error occurred while loading documents from CSV: {e}", exc_info=True)
    return documents

def main():
    """
    Main function to load documents from a CSV and build/update the FAISS index
    via the VectorStore.
    """
    logger.info("🚀 Starting FAISS index building process from CSV...")

    # Initialize the VectorStore. It will handle loading/creating the FAISS index internally.
    # If a FAISS index already exists, it will load it. Otherwise, it will be empty.
    vs = VectorStore() 

    # Load documents from the specified CSV file
    docs_to_add = load_knowledge_base_from_csv(CSV_PATH)

    if not docs_to_add:
        logger.warning("No documents loaded from CSV. FAISS index will not be built or updated.")
        return

    # Add documents to the VectorStore. This will embed them and add to the FAISS index.
    # The VectorStore class is responsible for persistence (saving the index and doc store).
    try:
        vs.add_documents(docs_to_add) # Assuming add_documents method can take a list
        logger.info(f"✅ Successfully added {len(docs_to_add)} documents to the VectorStore (FAISS index updated).")
        logger.info(f"FAISS index and document store saved/updated in '{settings.FAISS_INDEX_PATH}' and '{settings.FAISS_DOCS_PATH}'.")
    except Exception as e:
        logger.error(f"❌ Failed to add documents to VectorStore: {e}", exc_info=True)

    logger.info("🏁 FAISS index building process from CSV complete.")

if __name__ == '__main__':
    main()
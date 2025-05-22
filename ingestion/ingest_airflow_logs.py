import requests
import json
import os
import sys
import logging
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from datetime import datetime

# Add parent directory to sys.path to allow importing modules from 'agent'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agent.utils.log_preprocessor import extract_log_details, classify_log_category
from agent.rag.vector_store import VectorStore, Document # Import Document for creation
from agent.clients.github_client import GitHubClient
from agent.workflows.config import settings

# Configure logging for this module
logger = logging.getLogger(__name__)

# Load environment variables (should be done once at application start)
load_dotenv()

# Use settings for configuration
AIRFLOW_BASE_URL = settings.AIRFLOW_BASE_URL
AIRFLOW_USERNAME = settings.AIRFLOW_USERNAME.get_secret_value() # Get secret value
AIRFLOW_PASSWORD = settings.AIRFLOW_PASSWORD.get_secret_value() # Get secret value

# Output file for raw logs (optional, for debugging/record-keeping)
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "raw_ingested_airflow_logs.jsonl")

auth = HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
headers = {"Accept": "application/json"}

def _make_airflow_api_call(url: str) -> dict:
    """Helper function to make robust Airflow API calls."""
    try:
        response = requests.get(url, headers=headers, auth=auth, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - Response: {response.text}", exc_info=True)
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error occurred: {conn_err}. Is Airflow running at {url}?", exc_info=True)
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout error occurred: {timeout_err}", exc_info=True)
    except requests.exceptions.RequestException as req_err:
        logger.error(f"An unexpected error occurred: {req_err}", exc_info=True)
    return {}

def fetch_all_dags() -> list:
    """Fetches a list of all DAGs from Airflow."""
    url = f"{AIRFLOW_BASE_URL}/dags?limit=1000"
    data = _make_airflow_api_call(url)
    return data.get("dags", [])

def fetch_dag_runs(dag_id: str) -> list:
    """Fetches recent DAG runs for a specific DAG."""
    url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns?limit=100&order_by=-execution_date" # Fetch recent runs
    data = _make_airflow_api_call(url)
    return data.get("dag_runs", [])

def fetch_task_instances(dag_id: str, dag_run_id: str) -> list:
    """Fetches task instances for a specific DAG run."""
    url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances?limit=1000"
    data = _make_airflow_api_call(url)
    return data.get("task_instances", [])

def fetch_task_log(dag_id: str, dag_run_id: str, task_id: str) -> str:
    """Fetches the log content for a specific task instance."""
    # Airflow logs endpoint returns plain text, not JSON
    url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/1"
    try:
        response = requests.get(url, headers=headers, auth=auth, timeout=30)
        response.raise_for_status()
        return response.text # Raw text content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching log for {dag_id}.{task_id} [{dag_run_id}]: {e}", exc_info=True)
        return ""

def main():
    """
    Main ingestion function. Connects to Airflow API, fetches failed task logs,
    preprocesses them, and ingests them into the VectorStore.
    Optionally creates/updates GitHub issues for historical failures.
    """
    vs = VectorStore()
    # gh = GitHubClient() # Not needed for historical issue creation if using failure_notifier for live issues
    all_ingested_docs = []

    logger.info("🚀 Starting full Airflow historical log ingestion...")

    dags_fetched = 0
    runs_fetched = 0
    tasks_processed = 0
    failed_tasks_ingested = 0

    dags = fetch_all_dags()
    logger.info(f"Found {len(dags)} DAGs.")

    for dag in dags:
        dag_id = dag.get("dag_id")
        if not dag_id:
            logger.warning("Skipping DAG with no ID.")
            continue
        dags_fetched += 1
        logger.info(f"📘 Processing DAG: {dag_id}")

        dag_runs = fetch_dag_runs(dag_id)
        runs_fetched += len(dag_runs)
        logger.info(f"  Found {len(dag_runs)} runs for {dag_id}.")

        for run in dag_runs:
            run_id = run.get("dag_run_id")
            if not run_id:
                logger.warning(f"Skipping run with no ID for DAG {dag_id}.")
                continue
            
            task_instances = fetch_task_instances(dag_id, run_id)
            tasks_processed += len(task_instances)
            logger.info(f"    Found {len(task_instances)} task instances for run {run_id}.")

            for task in task_instances:
                task_id = task.get("task_id")
                state = task.get("state", "").lower()

                if not task_id:
                    logger.warning(f"Skipping task instance with no ID in run {run_id} of DAG {dag_id}.")
                    continue

                logger.debug(f"      🔍 {dag_id}.{task_id} [Run: {run_id}, State: {state}]...")

                if state != "failed": # Only ingest failed task logs
                    continue

                raw_log = fetch_task_log(dag_id, run_id, task_id)
                if not raw_log:
                    logger.warning(f"        ⚠️ No logs found for {dag_id}.{task_id} in run {run_id} — skipping.")
                    continue

                # Use the enhanced extract_log_details from log_preprocessor
                log_details = extract_log_details(raw_log)
                
                # Add Airflow-specific metadata to log_details for storage and retrieval
                log_details["dag_id"] = dag_id
                log_details["task_id"] = task_id
                log_details["dag_run_id"] = run_id
                log_details["source"] = "airflow_historical"
                
                # Classify the log to get a category label
                log_category = classify_log_category(log_details["full_log"])
                log_details["category"] = log_category # Add category to metadata

                # Create a Document for the VectorStore
                # The page_content is what gets embedded. Use the full cleaned log or error summary.
                # Metadata contains additional searchable/filterable attributes.
                doc_content = f"Airflow Task Failure: {log_details['error_summary']}\nDAG: {dag_id}, Task: {task_id}\nLog Snippet:\n{log_details['log_tail']}"

                document_to_add = Document(
                    page_content=doc_content,
                    metadata={
                        "dag_id": dag_id,
                        "task_id": task_id,
                        "dag_run_id": run_run_id, # Ensure this is the actual run ID
                        "category": log_category,
                        "error_summary": log_details["error_summary"],
                        "container_id": log_details["container_id"],
                        "source": "airflow_historical_logs",
                        "full_log_excerpt": log_details["full_log"][:settings.MAX_EMBEDDING_LOG_LENGTH], # Store a portion for LLM
                        "traceback_excerpt": log_details["traceback"][:settings.MAX_EMBEDDING_LOG_LENGTH], # Store portion of traceback
                    }
                )
                
                # Add the document to FAISS
                vs.add_document(document_to_add)
                all_ingested_docs.append(log_details) # Keep track of what was ingested
                failed_tasks_ingested += 1
                logger.info(f"        ✅ Ingested {dag_id}.{task_id} [{run_id}] (Category: {log_category})")

    # Optionally write all ingested raw details to a JSONL file
    if all_ingested_docs:
        try:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for log_data in all_ingested_docs:
                    f.write(json.dumps(log_data) + "\n")
            logger.info(f"\n✅ All ingested log details saved to {OUTPUT_FILE}.")
        except IOError as e:
            logger.error(f"Failed to write to output file {OUTPUT_FILE}: {e}", exc_info=True)

    logger.info(f"\n--- Ingestion Summary ---")
    logger.info(f"Total DAGs processed: {dags_fetched}")
    logger.info(f"Total DAG Runs fetched: {runs_fetched}")
    logger.info(f"Total Task Instances processed: {tasks_processed}")
    logger.info(f"Total Failed Task Logs ingested into VectorStore: {failed_tasks_ingested}")
    logger.info("🏁 Airflow historical log ingestion complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Set requests logger to WARNING to reduce verbosity
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    main()
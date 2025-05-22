import json
import logging
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

# Configure logging for this module
logger = logging.getLogger(__name__)

# Define file paths for the queue and processed logs
# These should ideally be configured via settings, but keeping them here for now
# consistent with previous examples if not already moved.
QUEUE_FILE = Path("./approval_queue.jsonl")
PROCESSED_LOG_FILE = Path("./processed_approvals.jsonl")

class ApprovalQueue:
    """
    Manages the queue of proposed actions awaiting approval and logs processed actions.
    Uses JSONL files for persistence.
    """
    def __init__(self):
        self.queue_file = QUEUE_FILE
        self.log_file = PROCESSED_LOG_FILE
        self._ensure_files_exist()

    def _ensure_files_exist(self):
        """Ensures that the queue and log files exist."""
        try:
            self.queue_file.touch(exist_ok=True)
            self.log_file.touch(exist_ok=True)
            logger.info(f"Ensured approval queue file exists: {self.queue_file}")
            logger.info(f"Ensured processed approvals log file exists: {self.log_file}")
        except IOError as e:
            logger.error(f"Failed to create queue files: {e}", exc_info=True)
            raise

    def _read_all_items(self, file_path: Path) -> List[Dict[str, Any]]:
        """Reads all valid JSON objects from a JSONL file."""
        items = []
        if not file_path.exists():
            return items

        with file_path.open("r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    items.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.error(f"Skipping malformed JSON line in {file_path} at line {line_num}: {e}. Line content: '{line.strip()}'")
                except Exception as e:
                    logger.error(f"An unexpected error reading line {line_num} from {file_path}: {e}", exc_info=True)
        return items

    def _write_all_items(self, file_path: Path, items: List[Dict[str, Any]]):
        """Writes a list of JSON objects back to a JSONL file, overwriting its content."""
        try:
            with file_path.open("w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
            logger.debug(f"Successfully wrote {len(items)} items to {file_path}")
        except IOError as e:
            logger.error(f"Failed to write to file {file_path}: {e}", exc_info=True)
            raise

    def exists(self, issue_number: int, key: str) -> bool:
        """
        Checks if a specific action (identified by issue_number and a unique key)
        is already in the approval queue.

        Args:
            issue_number (int): The GitHub issue number.
            key (str): A unique key representing the specific action (e.g., "airflow-123-dag_id-task_id",
                       "devops-456-k8s_cluster-restart_pod").

        Returns:
            bool: True if the action exists in the queue, False otherwise.
        """
        queue_items = self._read_all_items(self.queue_file)
        for item in queue_items:
            if item.get("issue_number") == issue_number and item.get("key") == key:
                logger.info(f"Item with key '{key}' for issue #{issue_number} already exists in queue.")
                return True
        logger.debug(f"Item with key '{key}' for issue #{issue_number} not found in queue.")
        return False

    def retry_count(self, issue_number: int, key: str) -> int:
        """
        Calculates how many times a specific action has been processed (retried)
        by checking the processed approvals log.

        Args:
            issue_number (int): The GitHub issue number.
            key (str): A unique key representing the specific action.

        Returns:
            int: The number of times this specific action has been processed, plus one
                 (since the first attempt counts as 1).
        """
        count = 0
        logged_items = self._read_all_items(self.log_file)
        for item in logged_items:
            if item.get("issue_number") == issue_number and item.get("key") == key:
                count += 1
        logger.info(f"Retry count for key '{key}' on issue #{issue_number}: {count + 1}")
        return count + 1 # +1 because the current attempt is the first if count is 0

    def submit(self, agent: str, issue_number: int, payload: Dict[str, Any], summary: str = "", key: Optional[str] = None):
        """
        Submits a proposed action to the approval queue.

        Args:
            agent (str): The name of the agent proposing the action (e.g., "airflow", "devops").
            issue_number (int): The GitHub issue number.
            payload (Dict[str, Any]): The specific data needed by the agent for execution.
            summary (str): A summary of the proposed action for display to approvers.
            key (Optional[str]): A unique identifier for the action. If None, one will be generated.
        """
        if key is None:
            # Generate a default key if not provided (less robust for complex cases)
            key = f"{agent}-{issue_number}-{payload.get('dag_id') or payload.get('target_component') or payload.get('resource')}"
            logger.warning(f"No key provided for submission. Generated default key: '{key}'. Consider providing explicit keys for better uniqueness.")

        entry = {
            "agent": agent,
            "issue_number": issue_number,
            "payload": payload,
            "summary": summary,
            "timestamp": datetime.now().isoformat(), # Add timestamp for auditing
            "key": key # Store the unique key
        }

        # Check if the item already exists before adding
        if self.exists(issue_number, key):
            logger.warning(f"Attempted to submit duplicate item with key '{key}' for issue #{issue_number}. Skipping.")
            return

        with self.queue_file.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"Submitted item with key '{key}' for issue #{issue_number} to approval queue.")

    def get_approved_items(self) -> List[Dict[str, Any]]:
        """
        Retrieves all items currently in the approval queue.
        In this system, all items in the queue are considered 'approved' and ready for processing
        by the ApprovalProcessor, as external approval (e.g., a GitHub comment) is assumed to have occurred.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing an approved action.
        """
        items = self._read_all_items(self.queue_file)
        logger.info(f"Retrieved {len(items)} items from the approval queue for processing.")
        return items

    def mark_as_processed(self, issue_number: int, key: str):
        """
        Moves an item from the active approval queue to the processed log file.

        Args:
            issue_number (int): The GitHub issue number of the item.
            key (str): The unique key of the item to be marked as processed.
        """
        current_queue_items = self._read_all_items(self.queue_file)
        updated_queue_items = []
        item_found_and_moved = False

        for item in current_queue_items:
            if item.get("issue_number") == issue_number and item.get("key") == key:
                # Add a processed timestamp to the item before logging it
                item["processed_timestamp"] = datetime.now().isoformat()
                # Append to processed log
                with self.log_file.open("a") as f:
                    f.write(json.dumps(item) + "\n")
                logger.info(f"Moved item with key '{key}' for issue #{issue_number} to processed log.")
                item_found_and_moved = True
            else:
                updated_queue_items.append(item)
        
        if item_found_and_moved:
            # Rewrite the queue file with the item removed
            self._write_all_items(self.queue_file, updated_queue_items)
            logger.info(f"Removed item with key '{key}' for issue #{issue_number} from active queue.")
        else:
            logger.warning(f"Item with key '{key}' for issue #{issue_number} not found in active queue for removal.")


# Example Usage (for testing the ApprovalQueue functionality)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Clean up old test files if they exist
    if QUEUE_FILE.exists():
        QUEUE_FILE.unlink()
    if PROCESSED_LOG_FILE.exists():
        PROCESSED_LOG_FILE.unlink()

    queue = ApprovalQueue()

    # Test 1: Submit an Airflow item
    print("\n--- Test 1: Submitting an Airflow item ---")
    airflow_payload = {
        "dag_id": "my_dag_1",
        "task_id": "task_a",
        "dag_run_id": "run_123",
        "action": "clear_and_retrigger"
    }
    airflow_key = "airflow-101-my_dag_1-task_a-run_123"
    queue.submit(
        agent="airflow",
        issue_number=101,
        payload=airflow_payload,
        summary="Retrigger my_dag_1/task_a",
        key=airflow_key
    )
    print(f"Item exists after submission (airflow): {queue.exists(101, airflow_key)}")
    print(f"Retry count (airflow, initial): {queue.retry_count(101, airflow_key)}")

    # Test 2: Submit a DevOps item
    print("\n--- Test 2: Submitting a DevOps item ---")
    devops_payload = {
        "target_component": "k8s_cluster_dev",
        "proposed_action": "restart_deployment_api"
    }
    devops_key = "devops-202-k8s_cluster_dev-restart_deployment_api"
    queue.submit(
        agent="devops",
        issue_number=202,
        payload=devops_payload,
        summary="Restart Kubernetes API deployment",
        key=devops_key
    )
    print(f"Item exists after submission (devops): {queue.exists(202, devops_key)}")
    print(f"Retry count (devops, initial): {queue.retry_count(202, devops_key)}")

    # Test 3: Try submitting a duplicate Airflow item (should be skipped)
    print("\n--- Test 3: Submitting duplicate Airflow item ---")
    queue.submit(
        agent="airflow",
        issue_number=101,
        payload=airflow_payload,
        summary="Retrigger my_dag_1/task_a (duplicate)",
        key=airflow_key
    )

    # Test 4: Get all approved items
    print("\n--- Test 4: Getting approved items ---")
    approved_items = queue.get_approved_items()
    print(f"Number of approved items: {len(approved_items)}")
    for item in approved_items:
        print(f"  - Agent: {item['agent']}, Issue: {item['issue_number']}, Key: {item['key']}")

    # Test 5: Mark Airflow item as processed
    print("\n--- Test 5: Marking Airflow item as processed ---")
    queue.mark_as_processed(101, airflow_key)
    print(f"Item exists after processing (airflow): {queue.exists(101, airflow_key)}")
    print(f"Retry count (airflow, after 1st processing): {queue.retry_count(101, airflow_key)}")

    # Test 6: Re-submit the same Airflow item (simulating a re-approval after retry)
    print("\n--- Test 6: Re-submitting Airflow item after processing ---")
    queue.submit(
        agent="airflow",
        issue_number=101,
        payload=airflow_payload,
        summary="Retrigger my_dag_1/task_a (retry)",
        key=airflow_key # Using the same key
    )
    print(f"Item exists after re-submission (airflow): {queue.exists(101, airflow_key)}")
    print(f"Retry count (airflow, after re-submission): {queue.retry_count(101, airflow_key)}")


    # Clean up test files at the end
    print("\n--- Cleaning up test files ---")
    if QUEUE_FILE.exists():
        QUEUE_FILE.unlink()
        print(f"Removed {QUEUE_FILE}")
    if PROCESSED_LOG_FILE.exists():
        PROCESSED_LOG_FILE.unlink()
        print(f"Removed {PROCESSED_LOG_FILE}")
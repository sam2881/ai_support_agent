import requests
import logging
import re
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any, List, Optional, Tuple

# Assuming 'settings' is imported from your project's config module
# This client expects 'settings' to be an object (e.g., Pydantic BaseSettings instance)
# with attributes like AIRFLOW_API_BASE, AIRFLOW_USER, AIRFLOW_PASS, etc.
from agent.workflows.config import settings # Ensure this path is correct for your project

logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class AirflowClientError(Exception):
    """Base exception for AirflowClient errors."""
    pass

class AirflowAPIError(AirflowClientError):
    """Exception raised for Airflow API errors (e.g., 4xx, 5xx responses)."""
    def __init__(self, message, status_code=None, response_text=None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text

class AirflowConnectionError(AirflowClientError):
    """Exception raised for network or connection issues."""
    pass


class AirflowClient:
    """
    A client for interacting with the Apache Airflow REST API.
    Designed for production use with features like sessions, timeouts, and retries.
    """
    DEFAULT_DAG_RUNS_LIMIT = 25
    DEFAULT_TASK_LOG_TRY_NUMBER = 1 # Usually, logs are fetched for a specific try
    DEFAULT_REQUEST_TIMEOUT = (10, 30) # (connect_timeout, read_timeout)
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_BACKOFF_FACTOR = 0.5


    def __init__(self):
        """
        Initializes the AirflowClient.
        - Sets up base URL, authentication, and headers from the 'settings' object.
        - Configures a requests.Session with timeouts and retry strategy.
        """
        if not hasattr(settings, 'AIRFLOW_API_BASE') or not settings.AIRFLOW_API_BASE:
            logger.error("AIRFLOW_API_BASE not configured in settings.")
            raise ValueError("AIRFLOW_API_BASE must be set in settings.")
        
        self.base_url = settings.AIRFLOW_API_BASE.rstrip("/")
        
        if not hasattr(settings, 'AIRFLOW_USER'):
            logger.error("AIRFLOW_USER not configured in settings.")
            raise ValueError("AIRFLOW_USER must be set in settings.")
        
        if not hasattr(settings, 'AIRFLOW_PASS'):
            logger.error("AIRFLOW_PASS not configured in settings.")
            raise ValueError("AIRFLOW_PASS must be set in settings.")

        try:
            # Pydantic's SecretStr has get_secret_value()
            airflow_password = settings.AIRFLOW_PASS.get_secret_value()
        except AttributeError:
            logger.warning("settings.AIRFLOW_PASS does not have get_secret_value(). Assuming it's a plain string (less secure).")
            airflow_password = settings.AIRFLOW_PASS

        self.auth = HTTPBasicAuth(settings.AIRFLOW_USER, airflow_password)
        
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Get timeout, retries, and backoff factor from settings, with defaults
        self.timeout = getattr(settings, 'AIRFLOW_REQUEST_TIMEOUT', self.DEFAULT_REQUEST_TIMEOUT)
        if not isinstance(self.timeout, tuple) or len(self.timeout) != 2:
            logger.warning(f"Invalid AIRFLOW_REQUEST_TIMEOUT format: {self.timeout}. Using default {self.DEFAULT_REQUEST_TIMEOUT}.")
            self.timeout = self.DEFAULT_REQUEST_TIMEOUT

        self.max_retries = getattr(settings, 'AIRFLOW_MAX_RETRIES', self.DEFAULT_MAX_RETRIES)
        self.retry_backoff_factor = getattr(settings, 'AIRFLOW_RETRY_BACKOFF_FACTOR', self.DEFAULT_RETRY_BACKOFF_FACTOR)

        self.session = self._create_session()
        logger.info(f"AirflowClient initialized for API base: {self.base_url}")

    def _create_session(self) -> requests.Session:
        """
        Creates a requests.Session with a retry mechanism.
        Retries are configured for common transient HTTP errors.
        """
        session = requests.Session()
        session.auth = self.auth
        session.headers.update(self.headers)

        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504], # Common transient server errors
            allowed_methods=["HEAD", "GET", "PUT", "POST", "DELETE", "OPTIONS", "PATCH"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Helper method to make requests to the Airflow API.
        Handles common request parameters, error handling, and logging.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.debug(f"Airflow API Request: {method.upper()} {url}, Params: {kwargs.get('params')}, Body: {kwargs.get('json')}")
        
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            logger.debug(f"Airflow API Response: {response.status_code} for {url}")
            return response
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            response_text = e.response.text if e.response is not None else "No response body"
            logger.error(f"Airflow API HTTPError: {status_code} for {url}. Response: {response_text}", exc_info=True)
            raise AirflowAPIError(
                f"Airflow API request failed for {url}: {e}",
                status_code=status_code,
                response_text=response_text
            ) from e
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Airflow API ConnectionError for {url}: {e}", exc_info=True)
            raise AirflowConnectionError(f"Connection error while trying to reach Airflow API at {url}: {e}") from e
        except requests.exceptions.Timeout as e:
            logger.error(f"Airflow API Timeout for {url}: {e}", exc_info=True)
            raise AirflowConnectionError(f"Request timed out for Airflow API at {url}: {e}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"Airflow API RequestException for {url}: {e}", exc_info=True)
            raise AirflowClientError(f"An unexpected error occurred during Airflow API request to {url}: {e}") from e

    def trigger_dag(self, dag_id: str, dag_run_id: Optional[str] = None, conf: Optional[Dict[str, Any]] = None, logical_date: Optional[str] = None, note: Optional[str] = None) -> Dict[str, Any]:
        """
        Triggers a new DAG run for the specified DAG ID.

        Args:
            dag_id: The ID of the DAG to trigger.
            dag_run_id: Optional. A unique ID for the DAG run. If not provided, Airflow generates one.
            conf: Optional. Configuration JSON object for the DAG run.
            logical_date: Optional. The logical date for the DAG run (ISO 8601).
            note: Optional. A note to add to the DAG run.

        Returns:
            A dictionary containing the API response from Airflow, typically details of the triggered run.
        
        Raises:
            AirflowAPIError: If the API call fails.
        """
        endpoint = f"dags/{dag_id}/dagRuns"
        payload: Dict[str, Any] = {"conf": conf or {}}
        if dag_run_id:
            payload["dag_run_id"] = dag_run_id
        if logical_date:
            payload["logical_date"] = logical_date 
        if note:
            payload["note"] = note
            
        response = self._request("POST", endpoint, json=payload)
        logger.info(f"Successfully triggered DAG '{dag_id}'. Run payload: {payload}")
        return response.json()

    def clear_task_instance(self, dag_id: str, dag_run_id: str, task_id: str, 
                            include_past: bool = False, include_future: bool = False,
                            include_upstream: bool = False, include_downstream: bool = False,
                            reset_dag_runs: bool = False, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clears task instances based on the provided parameters.
        Uses the standard Airflow API endpoint `/dags/{dag_id}/clearTaskInstances`.

        Args:
            dag_id: The DAG ID.
            dag_run_id: The DAG Run ID for which to clear tasks.
            task_id: The specific Task ID to clear (can be a list of task_ids).
            include_past: Clear past task instances.
            include_future: Clear future task instances.
            include_upstream: Clear upstream task instances.
            include_downstream: Clear downstream task instances.
            reset_dag_runs: Reset the DAG runs state.
            dry_run: If True, performs a dry run without actually clearing.

        Returns:
            A dictionary containing the API response from Airflow.
        
        Raises:
            AirflowAPIError: If the API call fails.
        """
        endpoint = f"dags/{dag_id}/clearTaskInstances"
        payload = {
            "dag_run_id": dag_run_id,
            "task_ids": [task_id] if isinstance(task_id, str) else task_id, 
            "include_past": include_past,
            "include_future": include_future,
            "include_upstream": include_upstream,
            "include_downstream": include_downstream,
            "reset_dag_runs": reset_dag_runs,
            "dry_run": dry_run,
        }
        
        response = self._request("POST", endpoint, json=payload)
        logger.info(f"Clear task instance(s) request for '{task_id}' in DAG '{dag_id}', Run ID '{dag_run_id}' processed. Dry run: {dry_run}.")
        return response.json() 

    def get_dag_runs(self, dag_id: str, limit: int = DEFAULT_DAG_RUNS_LIMIT, offset: int = 0, 
                     order_by: str = "-execution_date", states: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves DAG runs for a specific DAG ID, with pagination and filtering.

        Args:
            dag_id: The ID of the DAG.
            limit: The maximum number of DAG runs to retrieve.
            offset: The number of DAG runs to skip (for pagination).
            order_by: Field to order results by (e.g., "execution_date", "-start_date").
            states: Optional list of DAG run states to filter by (e.g., ["success", "failed"]).

        Returns:
            A list of dictionaries, where each dictionary represents a DAG run.
        """
        endpoint = f"dags/{dag_id}/dagRuns"
        params: Dict[str, Any] = {"limit": limit, "offset": offset, "order_by": order_by}
        if states:
            params["state"] = ",".join(states) 

        response = self._request("GET", endpoint, params=params)
        data = response.json()
        dag_runs = data.get("dag_runs", [])
        logger.info(f"Retrieved {len(dag_runs)} DAG runs for DAG '{dag_id}' with params: {params}.")
        return dag_runs

    def get_task_logs(self, dag_id: str, dag_run_id: str, task_id: str, task_try_number: int = DEFAULT_TASK_LOG_TRY_NUMBER, full_content: bool = False) -> Dict[str, Optional[str]]:
        """
        Fetches and parses logs for a specific task instance and try number.

        Args:
            dag_id: The DAG ID.
            dag_run_id: The DAG Run ID.
            task_id: The Task ID.
            task_try_number: The try number of the task instance.
            full_content: If True, requests full log content.

        Returns:
            A dictionary containing: full_log, log_excerpt, traceback, error_summary, container_id.
        """
        endpoint = f"dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{task_try_number}"
        params = {"full_content": str(full_content).lower()} 

        response = self._request("GET", endpoint, params=params)
        full_log = response.json().get("content", "")
        
        if not full_log:
            logger.warning(f"No log content found for {dag_id}/{dag_run_id}/{task_id}/try_{task_try_number}")
            return {
                "full_log": None, "log_excerpt": None, "traceback": None,
                "error_summary": "No log content found.", "container_id": None
            }

        lines = full_log.splitlines()
        log_excerpt = "\n".join(lines[-100:]) 

        traceback_text = None
        error_summary = None
        container_id = None

        tb_indices = [i for i, line in enumerate(lines) if "Traceback (most recent call last):" in line]
        if tb_indices:
            last_tb_start_index = tb_indices[-1]
            tb_block = []
            for i in range(last_tb_start_index, len(lines)):
                tb_block.append(lines[i])
                if len(tb_block) > 50: 
                    break
            traceback_text = "\n".join(tb_block)

        error_search_lines = lines[tb_indices[-1]:] if tb_indices else lines
        for line in reversed(error_search_lines): 
            if re.search(r"(?i)\b(error|exception|failed|failure|critical)\b", line) or \
               re.search(r"^\w+Error:", line): 
                if not line.strip().startswith("DEBUG") and not line.strip().startswith("INFO"): 
                    error_summary = line.strip()
                    break
        if not error_summary and lines: 
            error_summary = "No specific error line identified in excerpt; review full log."

        for line in lines:
            match = re.search(r"container_id=([a-f0-9]+)|(?:on host|in container)\s+([a-zA-Z0-9_-]+)", line, re.IGNORECASE)
            if match:
                container_id = match.group(1) or match.group(2)
                break
        
        logger.info(f"Successfully retrieved logs for {dag_id}/{dag_run_id}/{task_id}/try_{task_try_number}")
        return {
            "full_log": full_log,
            "log_excerpt": log_excerpt,
            "traceback": traceback_text,
            "error_summary": error_summary,
            "container_id": container_id,
        }

    def get_task_instance_details(self, dag_id: str, dag_run_id: str, task_id: str) -> Dict[str, Any]:
        """
        Retrieves details for a specific task instance.

        Args:
            dag_id: The DAG ID.
            dag_run_id: The DAG Run ID.
            task_id: The Task ID.

        Returns:
            A dictionary containing the task instance details.
        """
        endpoint = f"dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}"
        response = self._request("GET", endpoint)
        logger.info(f"Retrieved details for task instance: {dag_id}/{dag_run_id}/{task_id}")
        return response.json()

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # This example assumes you have a 'agent.workflows.config' module with a 'Settings' instance.
    # If running standalone, you'd need to mock or provide 'settings' differently.
    # For instance, you could create a dummy .env file and instantiate Settings here:
    #
    # with open(".env", "w") as f:
    #     f.write("AIRFLOW_API_BASE=http://localhost:8080/api/v1\n")
    #     f.write("AIRFLOW_USER=admin\n")
    #     f.write("AIRFLOW_PASS=admin\n")
    #     # Add other necessary settings for your 'Settings' class if they are required for instantiation
    #
    # from agent.workflows.config import Settings # Assuming your Settings class is here
    # settings = Settings() # This would load from .env if configured

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("Attempting to initialize AirflowClient using settings from agent.workflows.config...")
    
    try:
        # Ensure 'settings' is available and configured before creating an AirflowClient instance
        if 'settings' not in globals() or settings is None:
             raise NameError("The 'settings' object is not defined. Ensure it's imported from agent.workflows.config or mocked for testing.")

        client = AirflowClient() # Uses the globally imported 'settings'

        test_dag_id = "example_bash_operator" # Replace with a DAG ID in your Airflow
        
        logger.info(f"--- Testing get_dag_runs for DAG: {test_dag_id} ---")
        dag_runs = client.get_dag_runs(dag_id=test_dag_id, limit=2, states=["success", "failed"])
        if dag_runs:
            logger.info(f"Found {len(dag_runs)} runs. First run ID: {dag_runs[0].get('dag_run_id')}")
            
            test_dag_run_id = dag_runs[0].get('dag_run_id')
            test_task_id = "run_this_last" # Replace with an actual task_id from the DAG run

            if test_dag_run_id:
                logger.info(f"--- Testing get_task_instance_details for Task: {test_task_id} ---")
                task_details = client.get_task_instance_details(test_dag_id, test_dag_run_id, test_task_id)
                logger.info(f"Task instance state: {task_details.get('state')}")

                logger.info(f"--- Testing get_task_logs for Task: {test_task_id} ---")
                logs = client.get_task_logs(test_dag_id, test_dag_run_id, test_task_id, task_try_number=1)
                if logs.get("full_log"):
                    logger.info(f"Log excerpt (last 100 lines):\n{logs['log_excerpt'][:500]}...") 
                    logger.info(f"Extracted Error Summary: {logs['error_summary']}")
                    logger.info(f"Extracted Traceback (first 200 chars): {str(logs['traceback'])[:200]}...")
                    logger.info(f"Extracted Container ID: {logs['container_id']}")
                else:
                    logger.warning("No logs retrieved or log content was empty.")
        else:
            logger.warning(f"No DAG runs found for {test_dag_id} to proceed with further tests.")

    except NameError as ne:
        logger.error(f"Initialization Error: {ne}. Make sure 'settings' is correctly imported and configured.")
    except ValueError as ve:
        logger.error(f"Configuration Error: {ve}")
    except AirflowClientError as e:
        logger.error(f"Airflow Client operation failed: {e}")
        if isinstance(e, AirflowAPIError):
            logger.error(f"Status Code: {e.status_code}, Response: {e.response_text}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the example usage: {e}", exc_info=True)


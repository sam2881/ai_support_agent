import re
from typing import Dict, Any, Tuple

def extract_log_details(raw_log: str, tail_lines: int = 15) -> Dict[str, str]:
    """
    Processes raw Airflow logs to extract key information like full log,
    traceback, error summary, container ID, and a log tail.

    Args:
        raw_log (str): The raw log content fetched from Airflow.
        tail_lines (int): The number of lines to include in the log_tail.

    Returns:
        Dict[str, str]: A dictionary containing:
            - 'full_log': The cleaned full log content.
            - 'traceback': The extracted Python traceback (or a note if not found).
            - 'error_summary': The final error line from the traceback or log.
            - 'container_id': An extracted container/host ID if found.
            - 'log_tail': The last `tail_lines` of the cleaned log.
    """
    if not raw_log:
        return {
            "full_log": "",
            "traceback": "No log content.",
            "error_summary": "No log content.",
            "container_id": "Unavailable",
            "log_tail": "No log content."
        }

    # 1. Strip ANSI color codes
    clean_log = re.sub(r'\x1b\[.*?m', '', raw_log)

    # 2. Remove Airflow timestamps and process metadata prefixes
    # This regex is more robust to variations like "[2024-05-01, 12:00:00 +0000]" or different {{...}} contents
    clean_log = re.sub(r"\[\d{4}-\d{2}-\d{2},.*?\] \{\{.*?\}\} ", "", clean_log)
    
    # Normalize excessive newlines
    clean_log = re.sub(r'\n{3,}', '\n\n', clean_log).strip()

    traceback_content = "No traceback found."
    error_summary = "No specific error summary found."
    container_id = "Unavailable"

    # 3. Extract Traceback
    # This regex captures from "Traceback (most recent call last):" up to the next blank line
    # or the end of the log, or until a new log prefix indicates a new entry.
    traceback_match = re.search(r"Traceback \(most recent call last\):[\s\S]*?(?=\n\n|\n[A-Z][a-z]+Error:|\Z)", clean_log)
    if traceback_match:
        traceback_content = traceback_match.group(0).strip()
        # Remove the traceback from the main log if we want to keep it separate for LLM
        # However, for an LLM that reads the full log, keeping it in full_log is fine.
        # We extract it here primarily for a dedicated 'traceback' field.
    
    # 4. Extract Final Error Line (often the last line of the traceback)
    error_line_match = re.search(r"^\S*Error:.*$", clean_log, re.MULTILINE | re.IGNORECASE)
    if error_line_match:
        # Prioritize the last occurring error line
        all_error_lines = re.findall(r"^\S*Error:.*$", clean_log, re.MULTILINE | re.IGNORECASE)
        if all_error_lines:
            error_summary = all_error_lines[-1].strip()
    elif traceback_content != "No traceback found.":
        # If a traceback exists, try to get the last line of the traceback as error summary
        last_traceback_line = traceback_content.split('\n')[-1]
        if last_traceback_line:
            error_summary = last_traceback_line.strip()
    
    # Fallback for error summary if no specific error line was found but there's an exception message
    if "exception" in raw_log.lower() and "error" not in error_summary.lower():
         # Try to find a generic exception message
        exception_match = re.search(r"Exception: (.*)", raw_log, re.IGNORECASE)
        if exception_match:
            error_summary = f"Exception: {exception_match.group(1).strip()}"


    # 5. Extract Container ID (example pattern, adjust as needed)
    container_id_match = re.search(r"(?:container|pod|host) id:?\s*([a-zA-Z0-9_-]+)", raw_log, re.IGNORECASE)
    if container_id_match:
        container_id = container_id_match.group(1).strip()
    else:
        # Try to find common Kubernetes pod patterns
        pod_id_match = re.search(r"kubernetes/(pod|container)s/([a-zA-Z0-9_-]+)", raw_log)
        if pod_id_match:
            container_id = pod_id_match.group(2).strip()

    # 6. Extract Log Tail
    log_lines = clean_log.split('\n')
    log_tail = "\n".join(log_lines[-tail_lines:]).strip()

    return {
        "full_log": clean_log,
        "traceback": traceback_content,
        "error_summary": error_summary,
        "container_id": container_id,
        "log_tail": log_tail
    }

def classify_log_category(log_text: str) -> str:
    """
    Classifies the log into a general category based on keywords.
    This is a basic keyword-based classifier for initial high-level tagging.
    For more sophisticated routing, the LLM router is preferred.
    """
    log_lower = log_text.lower()

    if "xcom" in log_lower or "xcom_missing" in log_lower or "xcom_push" in log_lower or "xcom_pull" in log_lower:
        return "xcom_failure"
    elif "timeout" in log_lower or "timed out" in log_lower or "task exceeded timeout" in log_lower:
        return "timeout"
    elif "permission denied" in log_lower or "access denied" in log_lower or "unauthorized" in log_lower or "forbidden" in log_lower:
        return "access_issue"
    elif "connection refused" in log_lower or "connection reset" in log_lower or "host unreachable" in log_lower or "network error" in log_lower:
        return "connection_error"
    elif "kubernetes" in log_lower or "k8s" in log_lower or "pod" in log_lower or "container" in log_lower or "imagepullbackoff" in log_lower or "crashloopbackoff" in log_lower:
        return "infra"
    elif "dag not found" in log_lower or "dag doesn't exist" in log_lower:
        return "dag_not_found"
    # General errors that might indicate broader system issues
    elif "error" in log_lower or "fail" in log_lower or "exception" in log_lower:
        return "generic_error"
    
    return "unclassified" # Default if no specific keyword matches


# Example Usage for Testing
if __name__ == "__main__":
    print("--- Testing Log Preprocessor ---")

    # Example 1: Standard Airflow log with traceback and ANSI codes
    raw_log_1 = """
[2024-05-22, 10:00:00 +0000] {{taskinstance.py:1138}} INFO - Running command: ['airflow', 'tasks', 'run', 'my_dag', 'my_task', '2024-05-21T00:00:00+00:00', '--local', '--subdir', '/opt/airflow/dags']
[2024-05-22, 10:00:01 +0000] {{base_task_runner.py:144}} INFO - Job 123: Submitting command to executor.
[2024-05-22, 10:00:05 +0000] {{logging_mixin.py:112}} INFO - [2024-05-22 10:00:05,123] {__init__.py:45} INFO - This is a log line from a custom script.
\x1b[34m[2024-05-22, 10:00:06 +0000] {{python_callable.py:87}} ERROR - Traceback (most recent call last):
\x1b[0m  File "/usr/local/airflow/dags/my_dag.py", line 50, in my_task_callable
    result = 1 / 0
  File "/usr/local/lib/python3.8/site-packages/some_library/module.py", line 10, in divide_by_zero
    return a / b
ZeroDivisionError: division by zero
[2024-05-22, 10:00:07 +0000] {{taskinstance.py:1400}} ERROR - Task failed with exception
[2024-05-22, 10:00:07 +0000] {{models.py:1800}} INFO - Marking task as FAILED.
"""
    print("\n--- Raw Log 1 ---")
    print(raw_log_1)
    processed_1 = extract_log_details(raw_log_1)
    print("\n--- Processed Log 1 Details ---")
    print(f"Full Cleaned Log (start): {processed_1['full_log'][:200]}...")
    print(f"Traceback:\n{processed_1['traceback']}")
    print(f"Error Summary: {processed_1['error_summary']}")
    print(f"Container ID: {processed_1['container_id']}")
    print(f"Log Tail:\n{processed_1['log_tail']}")
    print(f"Category: {classify_log_category(processed_1['full_log'])}")

    # Example 2: Log with connection error and no explicit traceback
    raw_log_2 = """
[2024-05-22, 10:10:00 +0000] {{taskinstance.py:1138}} INFO - Attempting to connect to database...
[2024-05-22, 10:10:05 +0000] {{db_hook.py:200}} ERROR - Could not connect to host 'my_db.example.com': Connection refused.
[2024-05-22, 10:10:06 +0000] {{logging_mixin.py:112}} ERROR - Please check network connectivity and database service status.
"""
    print("\n--- Raw Log 2 ---")
    print(raw_log_2)
    processed_2 = extract_log_details(raw_log_2)
    print("\n--- Processed Log 2 Details ---")
    print(f"Full Cleaned Log (start): {processed_2['full_log'][:200]}...")
    print(f"Traceback:\n{processed_2['traceback']}")
    print(f"Error Summary: {processed_2['error_summary']}")
    print(f"Container ID: {processed_2['container_id']}")
    print(f"Log Tail:\n{processed_2['log_tail']}")
    print(f"Category: {classify_log_category(processed_2['full_log'])}")

    # Example 3: Log with XCom error
    raw_log_3 = """
[2024-05-22, 10:20:00 +0000] {{taskinstance.py:1138}} INFO - Starting task 'process_data'.
[2024-05-22, 10:20:02 +0000] {{xcom.py:100}} ERROR - XComArg result is missing. The previous task 'load_data' did not push XCom value.
[2024-05-22, 10:20:03 +0000] {{taskinstance.py:1400}} ERROR - Task failed due to XCom error.
"""
    print("\n--- Raw Log 3 ---")
    print(raw_log_3)
    processed_3 = extract_log_details(raw_log_3)
    print("\n--- Processed Log 3 Details ---")
    print(f"Full Cleaned Log (start): {processed_3['full_log'][:200]}...")
    print(f"Traceback:\n{processed_3['traceback']}")
    print(f"Error Summary: {processed_3['error_summary']}")
    print(f"Container ID: {processed_3['container_id']}")
    print(f"Log Tail:\n{processed_3['log_tail']}")
    print(f"Category: {classify_log_category(processed_3['full_log'])}")

    # Example 4: Log with container ID and no specific error line at the end
    raw_log_4 = """
[2024-05-22, 10:30:00 +0000] {{taskinstance.py:1138}} INFO - Running on container id: abcdef123456
[2024-05-22, 10:30:01 +0000] {{base_task_runner.py:144}} INFO - Pod 'my-app-pod-xyz' status: CrashLoopBackOff.
[2024-05-22, 10:30:02 +0000] {{logging_mixin.py:112}} INFO - Image pull failed for gcr.io/my-project/my-image:latest.
"""
    print("\n--- Raw Log 4 ---")
    print(raw_log_4)
    processed_4 = extract_log_details(raw_log_4)
    print("\n--- Processed Log 4 Details ---")
    print(f"Full Cleaned Log (start): {processed_4['full_log'][:200]}...")
    print(f"Traceback:\n{processed_4['traceback']}")
    print(f"Error Summary: {processed_4['error_summary']}")
    print(f"Container ID: {processed_4['container_id']}")
    print(f"Log Tail:\n{processed_4['log_tail']}")
    print(f"Category: {classify_log_category(processed_4['full_log'])}")
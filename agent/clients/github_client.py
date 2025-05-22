import logging
from github import Github, GithubException
import logging
from github import Github, GithubException
from agent.workflows.config import settings
# Assuming settings are correctly configured as in your original code
# from agent.workflows.config import settings

# Placeholder for settings if you run this snippet standalone
# Replace with your actual token and repo, or ensure they are in your environment
class SettingsPlaceholder:


    token = settings.GITHUB_TOKEN
    repo_name = settings.GITHUB_REPO

settings = SettingsPlaceholder() # Comment this out if using your own settings module

logger = logging.getLogger(__name__)
# Configure basic logging if running standalone for testing
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GitHubClient:
    def __init__(self):
        """
        Initializes the GitHub client, authenticates, and connects to the specified repository.
        Reads token and repo_name from the settings.
        """
        try:
            token = settings.GITHUB_TOKEN
            repo_name = settings.GITHUB_REPO

            if not token or not repo_name or token == "YOUR_token" or repo_name == "YOUR_USERNAME/YOUR_REPONAME":
                logger.error("❌ Missing or placeholder token or repo_name in settings.")
                logger.error("Please replace placeholder values or ensure they are correctly set in your environment/config.")
                raise ValueError("token and repo_name must be set to valid values.")

            self.gh = Github(token)
            self.repo = self.gh.get_repo(repo_name)
            logger.info(f"✅ Connected to GitHub repo: {self.repo.full_name}")

        except GithubException as ge:
            logger.error(f"❌ GitHub API Error: Failed to authenticate or fetch the repository '{repo_name}'. Status: {ge.status}, Data: {ge.data}", exc_info=True)
            raise
        except Exception as e:
            logger.error("❌ Initialization of GitHubClient failed.", exc_info=True)
            raise

    def create_issue(self, title, body, labels=None):
        """
        Creates a new GitHub issue or reopens an existing closed one if found with the same title.
        
        Args:
            title (str): The title of the issue.
            body (str): The body content of the issue.
            labels (list, optional): A list of labels to apply to the issue. Defaults to None.
            
        Returns:
            github.Issue.Issue or None: The created or existing issue object, or None if creation fails.
        """
        try:
            existing = self.find_existing_issue(title)
            if existing:
                if existing.state == "closed":
                    logger.info(f"♻️ Issue '#{existing.number} - {existing.title}' found closed. Reopening.")
                    existing.edit(state="open")
                    # Add a comment to indicate recurrence and point to new details if body structure allows
                    # For example, if the new body contains specific new log excerpts.
                    # existing.create_comment("🚨 This issue has recurred. See updated details in the issue description if modified, or new log link.")
                else:
                    logger.info(f"🔁 Issue '#{existing.number} - {existing.title}' already exists and is open.")
                return existing

            issue = self.repo.create_issue(
                title=title,
                body=body,
                labels=labels or []
            )
            logger.info(f"🐛 Created new GitHub issue: #{issue.number} - {title}")
            return issue

        except GithubException as ge:
            logger.error(f"❌ GitHub API error during issue creation for title '{title}'. Status: {ge.status}, Data: {ge.data}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error during issue creation for title '{title}'", exc_info=True)
            return None

    def create_descriptive_issue(self, dag_id, task_id, dag_run_id, log_excerpt, airflow_log_link, labels=None):
        """
        Creates a richly formatted GitHub issue for a DAG failure, including an Airflow log link
        and details structured to aid an LLM in routing the issue.
        
        Args:
            dag_id (str): The ID of the DAG.
            task_id (str): The ID of the failed task.
            dag_run_id (str): The ID of the DAG run.
            log_excerpt (str): A string containing relevant lines from the Airflow log.
            airflow_log_link (str): A direct URL to the full Airflow logs for this task instance.
            labels (list, optional): A list of labels to apply. Defaults to None.
            
        Returns:
            github.Issue.Issue or None: The created or existing issue object, or None if creation fails.
        """
        # Construct a unique and informative title
        title = f"🔥 Airflow Alert: DAG `{dag_id}` - Task `{task_id}` Failed (Run: {dag_run_id})"

        # Extract the most relevant error message and last N lines from the log excerpt
        last_lines = "\n".join(log_excerpt.strip().splitlines()[-20:]) # Show last 20 lines
        main_error = self.extract_error_message(log_excerpt)

        # Format the GitHub issue body with Markdown for clarity and LLM guidance
        body = f"""
### ❌ Automated Alert: Airflow DAG Task Failure

An issue has been automatically detected in an Apache Airflow DAG run. Analysis and action may be required.

- **DAG ID:** `{dag_id}`
- **Task ID:** `{task_id}`
- **Run ID:** `{dag_run_id}`
- **Status:** `failed`
- **🔗 Full Airflow Log:** [{airflow_log_link}]({airflow_log_link})  _(Review for complete context)_

---

### 🕵️‍♂️ Preliminary Diagnostics from Log Excerpt

**Key Error Message Detected:**
```text
{main_error}
```

**Recent Log Output (Last 20 lines of excerpt):**
```log
{last_lines}
```

---

### 🤖 LLM Triage & Assignment Guidance

**Objective for LLM Agent:**
Please analyze the **full Airflow log** (linked above) and the provided diagnostic details to:
1.  **Identify Root Cause:** Determine the most probable cause of this task failure (e.g., data issue, code error, external system failure, resource exhaustion).
2.  **Recommend Next Agent/Team:** Based on the root cause, suggest the most appropriate team or individual agent to investigate and resolve this issue (e.g., Data Engineering, Platform Operations, Application Support Team, specific SME).
3.  **Suggest Initial Actions (Optional):** If possible, propose 1-2 initial diagnostic questions or troubleshooting steps for the assigned agent.

**Output:** Please provide your analysis by **adding a comment to this GitHub issue**. You may also update labels or assignees if you have the necessary permissions and confidence.

---

🔧 *This issue was auto-generated by the AI-Powered Monitoring & Alerting System.*
        """
        return self.create_issue(title=title, body=body, labels=labels)

    def extract_error_message(self, log_text: str) -> str:
        """
        Extracts a concise and relevant error message from the log text.
        It prioritizes the latest full traceback, then specific error lines.
        
        Args:
            log_text (str): The log content as a string.
            
        Returns:
            str: The extracted error message or a fallback message.
        """
        lines = log_text.strip().splitlines()
        
        # Attempt to find and extract the last traceback
        traceback_block = []
        traceback_started = False
        for i in range(len(lines) - 1, -1, -1): # Iterate backwards
            line_stripped = lines[i].strip()
            if "Traceback (most recent call last):" in line_stripped:
                traceback_started = True
                traceback_block.append(lines[i]) # Add the "Traceback..." line itself
                # Extract up to N lines of the traceback above this point
                # This captures the error message and a few lines of the stack
                traceback_context = lines[max(0, i - 7) : i] # Get some lines before traceback start
                traceback_block.extend(reversed(traceback_context)) # Add context before
                break # Found the start of the latest traceback
            if traceback_started:
                 # Prepend lines to maintain order, limit total traceback lines extracted
                if len(traceback_block) < 15: # Max 15 lines for the traceback block
                    traceback_block.append(lines[i])
                else:
                    break # Limit reached
        
        if traceback_block:
            # Reverse to original order and join
            return "\n".join(reversed(traceback_block)).strip()

        # Fallback: Look for lines containing common error keywords from the end
        error_keywords = ["error", "exception", "critical", "fatal", "failed", "failure"]
        relevant_error_lines = []
        for line in reversed(lines):
            if any(keyword in line.lower() for keyword in error_keywords):
                relevant_error_lines.append(line)
                if len(relevant_error_lines) >= 5: # Get up to 5 such lines
                    break
        
        if relevant_error_lines:
            return "\n".join(reversed(relevant_error_lines)).strip()
            
        # If no specific error found, return last few lines as a generic message
        if lines:
            return "No specific error pattern found. Last few lines of excerpt:\n" + "\n".join(lines[-5:])
        
        return "Unknown failure: Log excerpt was empty or no error pattern detected. Please review the full log."

    def find_existing_issue(self, title: str):
        """
        Finds an existing issue (open or closed) with an exact matching title.
        Uses GitHub search API for potentially better efficiency.
        
        Args:
            title (str): The title to search for.
            
        Returns:
            github.Issue.Issue or None: The found issue object, or None if not found or error.
        """
        try:
            # Sanitize title for search query (though exact match is less problematic)
            # query_title = title.replace('"', '\\"') # Basic sanitization for quotes
            
            # Search for issues with this exact title in the current repository
            # The `in:title` qualifier ensures we only match titles.
            # Using quotes around the title enforces an exact phrase match.
            query = f'repo:"{self.repo.full_name}" is:issue in:title "{title}"'
            
            issues = self.gh.search_issues(query=query)
            
            # `search_issues` returns a PaginatedList. We need the first exact match.
            for issue in issues:
                if issue.title.strip().lower() == title.strip().lower():
                    logger.debug(f"Found existing issue via search: #{issue.number} - {issue.title}")
                    return issue
            logger.debug(f"No existing issue found via search for title: {title}")
            return None
        except GithubException as ge:
            # Handle cases like rate limiting or invalid search query
            logger.warning(f"⚠️ GitHub API error during issue search for '{title}': {ge.data}. Falling back to get_issues.", exc_info=False)
        except Exception as e:
            logger.warning(f"⚠️ Unexpected error during GitHub issue search for '{title}'. Falling back to get_issues.", exc_info=True)
        
        # Fallback to iterating all issues if search fails or as a less efficient alternative
        # This can be slow on repositories with many issues.
        try:
            logger.debug(f"Falling back to repo.get_issues() for title: {title}")
            issues = self.repo.get_issues(state="all") # Check both open and closed
            for issue in issues:
                if issue.title.strip().lower() == title.strip().lower():
                    logger.debug(f"Found existing issue via get_issues: #{issue.number} - {issue.title}")
                    return issue
            logger.debug(f"No existing issue found via get_issues for title: {title}")
        except GithubException as ge_get:
            logger.error(f"❌ GitHub API error during fallback get_issues for '{title}': {ge_get.data}", exc_info=True)
        except Exception as e_get:
            logger.error(f"❌ Unexpected error during fallback get_issues for '{title}'", exc_info=True)
        return None


    def get_issue(self, issue_number: int):
        """
        Retrieves a specific issue by its number.
        
        Args:
            issue_number (int): The number of the issue to fetch.
            
        Returns:
            github.Issue.Issue or None: The issue object, or None if not found or error.
        """
        try:
            issue = self.repo.get_issue(number=issue_number)
            logger.info(f"✅ Fetched issue #{issue.number} - {issue.title}")
            return issue
        except GithubException as ge:
            logger.error(f"❌ GitHub API error fetching issue #{issue_number}: {ge.data}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching issue #{issue_number}", exc_info=True)
            return None

    def close_issue(self, issue_number: int, comment: str = None):
        """
        Closes a GitHub issue, optionally adding a comment before closing.
        
        Args:
            issue_number (int): The number of the issue to close.
            comment (str, optional): A comment to add before closing. Defaults to None.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            issue = self.repo.get_issue(number=issue_number)
            if issue.state == "closed":
                logger.info(f"ℹ️ Issue #{issue_number} is already closed.")
                return True
            if comment:
                issue.create_comment(comment)
                logger.info(f"💬 Commented on issue #{issue_number} before closing.")
            issue.edit(state="closed")
            logger.info(f"✅ Closed GitHub issue #{issue_number}")
            return True
        except GithubException as ge:
            logger.error(f"❌ GitHub API error closing issue #{issue_number}: {ge.data}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error closing issue #{issue_number}", exc_info=True)
            return False

    def add_comment(self, issue_number: int, comment: str):
        """
        Adds a comment to a specific GitHub issue.
        
        Args:
            issue_number (int): The number of the issue to comment on.
            comment (str): The comment text.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # repo_owner and repo_name are not needed as self.repo is already the specific repo object
        try:
            issue = self.repo.get_issue(number=issue_number)
            issue.create_comment(comment)
            logger.info(f"💬 Commented on issue #{issue_number}")
            return True
        except GithubException as ge:
            logger.error(f"❌ GitHub API error commenting on issue #{issue_number}: {ge.data}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error commenting on issue #{issue_number}", exc_info=True)
            return False

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # --- IMPORTANT: Configure logging to see output ---
    logging.basicConfig(
        level=logging.INFO, # Use logging.DEBUG for more verbose output from the client
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting GitHubClient example...")

    try:
        # Ensure settings.token and settings.repo_name are correctly set
        # For example, you might load them from environment variables in your actual agent.workflows.config
        # Or, for this test, replace the SettingsPlaceholder values directly.
        # settings.token = "ghp_YOUR_FINE_GRAINED_TOKEN" # Replace if using placeholder
        # settings.repo_name = "YourUsername/YourTestRepo"   # Replace if using placeholder

        github_client = GitHubClient()

        # --- Mock Airflow context for the example ---
        dag_id = "daily_sales_processing_v3"
        task_id = "transform_sales_data"
        # For dag_run_id, it's often the logical date or a unique run identifier
        dag_run_id = "scheduled__2025-05-22T14:30:00+00:00" 
        
        # Construct an example Airflow log link (adapt to your Airflow's URL structure)
        # This often involves the Airflow webserver base URL, dag_id, task_id, and execution_date (or dag_run_id)
        airflow_webserver_base_url = "http://localhost:8080" # Replace with your Airflow webserver URL
        # Note: Airflow's UI log link might use execution_date. If dag_run_id is different, adjust accordingly.
        # For simplicity, assuming dag_run_id can be used or mapped to execution_date for the link.
        log_link = (f"{airflow_webserver_base_url}/dags/{dag_id}/grid?"
                    f"task_id={task_id}&dag_run_id={dag_run_id}&tab=logs")


        # Simulate a log excerpt (in a real scenario, you'd fetch this from Airflow logs)
        log_excerpt_with_traceback = """
[2025-05-22 14:35:00,123] {{taskinstance.py:1892}} INFO - Starting task: transform_sales_data
[2025-05-22 14:35:01,456] {{s3hook.py:456}} INFO - Downloading sales_data.csv from S3.
[2025-05-22 14:35:02,789] {{python_operator.py:175}} INFO - Executing Python callable.
[2025-05-22 14:35:03,123] {{script.py:55}} INFO - Processing 1000 records.
[2025-05-22 14:35:03,456] {{script.py:78}} ERROR - Encountered an error processing a record.
Traceback (most recent call last):
  File "/opt/airflow/dags/scripts/transform_sales.py", line 75, in process_record
    processed_value = int(record.get('value')) / int(record.get('divisor'))
ZeroDivisionError: division by zero
[2025-05-22 14:35:03,458] {{taskinstance.py:1943}} ERROR - Task failed with exception
Traceback (most recent call last):
  File "/opt/airflow/venv/lib/python3.9/site-packages/airflow/operators/python.py", line 171, in execute
    return_value = self.python_callable(*self.op_args, **self.op_kwargs)
  File "/opt/airflow/dags/scripts/transform_sales.py", line 120, in transform_data_callable
    raise e
  File "/opt/airflow/dags/scripts/transform_sales.py", line 75, in process_record
    processed_value = int(record.get('value')) / int(record.get('divisor'))
ZeroDivisionError: division by zero
        """
        
        labels_for_issue = ["bug", "airflow-failure", "data-pipeline", "needs-triage-llm"]

        logger.info(f"Attempting to create/update issue for DAG: {dag_id}, Task: {task_id}")
        issue = github_client.create_descriptive_issue(
            dag_id=dag_id,
            task_id=task_id,
            dag_run_id=dag_run_id,
            log_excerpt=log_excerpt_with_traceback,
            airflow_log_link=log_link,
            labels=labels_for_issue
        )

        if issue:
            logger.info(f"✅ Successfully processed issue. URL: {issue.html_url}")
            
            # Example: Add a follow-up comment (simulating LLM or another agent)
            # if not issue.comments: # Only comment if it's a newly created issue or has no comments
            #     github_client.add_comment(
            #         issue_number=issue.number,
            #         comment="🤖 LLM Analysis: Root cause appears to be a `ZeroDivisionError` in `transform_sales.py` line 75. "
            #                 "Suggest assigning to Data Engineering Team. Initial check: Verify input data for 'divisor' field being zero."
            #     )
            
            # Example: Close the issue (if it were resolved)
            # github_client.close_issue(issue.number, "Resolved: Input data corrected, divisor will no longer be zero.")
        else:
            logger.error("❌ Failed to create or update GitHub issue.")

    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
    except GithubException as ge:
        logger.error(f"A GitHub API error occurred: {ge.data}", exc_info=False) # exc_info=False as client logs it
    except Exception as e:
        logger.error(f"An unexpected error occurred in the example usage: {e}", exc_info=True)


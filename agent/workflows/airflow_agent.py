# agent/workflows/agents/airflow_agent.py
import logging
import json
from datetime import datetime
from typing import Dict, Any, List

from agent.clients.github_client import GitHubClient
from agent.clients.airflow_client import AirflowClient
from agent.rag.vector_store import VectorStore # Still imported for type hinting, though not directly queried
from agent.neo4j.graph_client import GraphClient # Still imported for type hinting, though not directly queried
from agent.models.prompts import SOLUTION_PROMPT # Assuming this prompt is suitable
from agent.workflows.config import settings
from agent.utils.queue_utils import ApprovalQueue

# Import the correct LLM library based on settings.LLM_MODEL
# Using google.generativeai if settings.LLM_MODEL indicates a Google model (e.g., "gemini-pro")
# Otherwise, default to langchain_openai for "gpt-4" etc.
# This assumes you have the appropriate library installed (e.g., 'google-generativeai' or 'openai')
if "gemini" in settings.LLM_MODEL:
    import google.generativeai as genai
    # Ensure GOOGLE_API_KEY is set in your .env
    genai.configure(api_key=settings.GOOGLE_API_KEY.get_secret_value())
    LLM_CLIENT = genai.GenerativeModel(settings.LLM_MODEL)
    logger.info(f"Initialized Google Generative AI client with model: {settings.LLM_MODEL}")
else:
    from langchain_openai import ChatOpenAI
    LLM_CLIENT = ChatOpenAI(
        model=settings.LLM_MODEL, # Use the model from settings
        temperature=0,
        openai_api_key=settings.OPENAI_API_KEY.get_secret_value() # Get secret value
    )
    logger.info(f"Initialized Langchain ChatOpenAI client with model: {settings.LLM_MODEL}")

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class AirflowAgent:
    def __init__(self):
        """
        Initializes the AirflowAgent with necessary clients and utilities.
        """
        self.gh = GitHubClient() # GitHub client
        self.af = AirflowClient() # Airflow client
        # VectorStore and GraphClient instances are now managed by MainAgent
        # self.vstore = VectorStore() # No longer directly queried by sub-agents
        # self.graph = GraphClient() # No longer directly queried by sub-agents
        self.queue = ApprovalQueue() # For managing approval workflows

    def handle(self, issue: Any, log_data: Dict[str, str], enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles an Airflow-related GitHub issue by analyzing logs and RAG context,
        proposing a remediation, and submitting it for approval.

        Args:
            issue (Any): The GitHub issue object (from PyGithub).
            log_data (Dict[str, str]): Parsed Airflow log details (full_log, error_summary, etc.)
                                        including 'dag_id', 'task_id', 'dag_run_id'.
            enrichment_data (Dict[str, Any]): Context from RAG sources (VectorStore, Neo4j)
                                              passed by the MainAgent.

        Returns:
            Dict[str, Any]: A dictionary containing the status of the agent's operation,
                            including proposed actions and LLM summary.
        """
        issue_number = issue.number
        issue_text = f"{issue.title}\n{issue.body or ''}"
        full_log = log_data.get("full_log", "")
        
        # Extract Airflow specific details from log_data (parsed by MainAgent)
        dag_id = log_data.get("dag_id")
        task_id = log_data.get("task_id")
        dag_run_id = log_data.get("dag_run_id")

        if not dag_id or not task_id:
            comment = "⚠️ Airflow Agent: Could not identify DAG ID or Task ID from the issue. Cannot propose specific remediation."
            self.gh.add_comment(issue_number, comment)
            logger.warning(f"Issue #{issue_number}: Missing DAG ID or Task ID for Airflow remediation.")
            return {"status": "skipped", "llm_summary": comment, "auto_approved": False}

        try:
            # Prepare RAG context for the LLM prompt
            faiss_context = self._format_faiss_docs(enrichment_data.get("faiss_docs", []))
            graph_similar_issues = json.dumps(enrichment_data.get("similar_issues_graph", []), indent=2)
            graph_related_solutions = json.dumps(enrichment_data.get("related_neo4j_solutions", []), indent=2)

            # Construct the LLM prompt using the SOLUTION_PROMPT template
            prompt_content = SOLUTION_PROMPT.format(
                issue_title=issue.title,
                issue_body=issue.body,
                full_log=full_log,
                dag_id=dag_id,
                task_id=task_id,
                dag_run_id=dag_run_id,
                faiss_solutions=faiss_context,
                graph_similar_issues=graph_similar_issues,
                graph_related_solutions=graph_related_solutions
            )

            # Call LLM and parse the response
            parser = JsonOutputParser()
            
            # Using the dynamically selected LLM client
            if "gemini" in settings.LLM_MODEL:
                # For Google Generative AI (Gemini)
                response = LLM_CLIENT.generate_content(prompt_content)
                llm_response_content = response.candidates[0].content.parts[0].text
            else:
                # For Langchain (OpenAI, etc.)
                messages = [
                    SystemMessage(content="You are a helpful assistant that provides solutions in JSON format."),
                    HumanMessage(content=prompt_content),
                ]
                llm_response_content = LLM_CLIENT.invoke(messages).content

            parsed = parser.parse(llm_response_content)
            
            # Ensure parsed output has the expected keys, provide defaults if missing
            proposed_action = parsed.get("proposed_action", "No specific action proposed.")
            reason = parsed.get("reason", "No reason provided by LLM.")
            confidence_score = parsed.get("confidence_score", 0.0)
            
            # Use original DAG/Task/Run IDs for retrigger
            summary = (
                f"🤖 **Airflow Agent Proposed Remediation**:\n\n"
                f"**Proposed Action**: {proposed_action}\n"
                f"**Target**: DAG `{dag_id}`, Task `{task_id}`"
                f"{f', Run ID `{dag_run_id}`' if dag_run_id else ''}\n"
                f"**Reasoning**: {reason}\n"
                f"**Confidence Score**: {confidence_score:.2f} (0.0-1.0)\n\n"
                f"Waiting for manual approval to proceed with execution."
            )
            
            self.gh.add_comment(issue_number, summary)
            logger.info(f"Issue #{issue_number}: Proposed remediation posted.")

            # Step 4: Submit for approval
            # The key for uniqueness in the queue should be stable
            queue_key = f"{dag_id}-{task_id}-{dag_run_id}"
            if not self.queue.exists(issue_number, queue_key): # Check existence using combined key
                retry_count = self.queue.retry_count(issue_number, queue_key)
                self.queue.submit(
                    agent="airflow",
                    issue_number=issue_number,
                    payload={
                        "dag_id": dag_id,
                        "task_id": task_id,
                        "dag_run_id": dag_run_id,
                        "retry_count": retry_count,
                        "proposed_action": proposed_action,
                        "reason": reason,
                        "confidence_score": confidence_score,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    },
                    summary=summary
                )
                logger.info(f"Issue #{issue_number}: Action submitted to approval queue.")
            else:
                logger.info(f"Issue #{issue_number}: Action already in approval queue. Skipping re-submission.")

            return {
                "agent": "airflow",
                "status": "awaiting_approval",
                "dag_id": dag_id,
                "task_id": task_id,
                "dag_run_id": dag_run_id,
                "llm_summary": summary,
                "auto_approved": False # Always requires approval for execution
            }

        except json.JSONDecodeError as e:
            error_msg = f"LLM response was not valid JSON. Error: {e}\nRaw response: {llm_response_content[:500]}..."
            logger.error(f"❌ AirflowAgent JSON parsing failed for issue #{issue_number}: {error_msg}", exc_info=True)
            self.gh.add_comment(issue_number, f"⚠️ Airflow Agent: Failed to parse LLM response. {error_msg}")
            return {"agent": "airflow", "status": "failed", "error": error_msg}
        except Exception as e:
            error_msg = f"AirflowAgent encountered an unexpected error: {e}"
            logger.error(f"❌ {error_msg} for issue #{issue_number}", exc_info=True)
            self.gh.add_comment(issue_number, f"⚠️ Airflow Agent: Remediation process failed. Error: `{e}`")
            return {"agent": "airflow", "status": "failed", "error": str(e)}

    def execute_remediation(self, issue_number: int, dag_id: str, task_id: str, dag_run_id: str = "") -> Dict[str, Any]:
        """
        Executes the proposed Airflow remediation (clearing/retriggering a task).
        This method would typically be called by the webhook handler upon approval.

        Args:
            issue_number (int): The GitHub issue number.
            dag_id (str): The ID of the Airflow DAG.
            task_id (str): The ID of the Airflow task.
            dag_run_id (str): Optional. The specific DAG run ID to clear/retrigger.

        Returns:
            Dict[str, Any]: Status of the execution.
        """
        logger.info(f"Attempting to execute remediation for Issue #{issue_number}: Clearing task {task_id} in DAG {dag_id} (Run: {dag_run_id or 'latest'}).")
        msg = ""
        try:
            # AirflowClient's clear_task_instance is the correct method for retriggering
            # by marking it as 'None' or 're-run'.
            # Note: Airflow's behavior for clearing task instances can be complex.
            # This generally marks the task instance as 'up_for_retry' or 'scheduled'
            # causing the scheduler to pick it up again.
            self.af.clear_task_instance(dag_id, dag_run_id, task_id)
            
            msg = (
                f"✅ **Airflow Task Retriggered**:\n"
                f"DAG `{dag_id}`, task `{task_id}` has been marked for re-run"
                f"{f' for run `{dag_run_id}`' if dag_run_id else ''}.\n"
                f"Please monitor Airflow UI for status."
            )
            logger.info(f"Successfully retriggered task for issue #{issue_number}.")
            # Close the GitHub issue after successful execution
            # self.gh.close_issue(issue_number, msg) # This is now handled by MainAgent/Webhook handler
            
            return {"status": "resolved", "message": msg, "auto_approved": True}
        except Exception as e:
            msg = f"❌ **Airflow Remediation Failed**:\n" \
                  f"Could not retrigger DAG `{dag_id}`, task `{task_id}`" \
                  f"{f' for run `{dag_run_id}`' if dag_run_id else ''}.\n" \
                  f"Error: `{e}`\n" \
                  f"Please attempt manual retrigger or investigate further."
            logger.error(f"Failed to retrigger task for issue #{issue_number}: {e}", exc_info=True)
            self.gh.add_comment(issue_number, msg) # Post failure message to GitHub
            return {"status": "failed", "message": msg, "auto_approved": False}

    def _format_faiss_docs(self, faiss_docs: List[Any]) -> str:
        """
        Helper to format FAISS documents into a string for the LLM prompt.
        """
        if not faiss_docs:
            return "No relevant historical solutions found."

        formatted_docs = []
        for i, doc in enumerate(faiss_docs):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            solution = metadata.get("solution", metadata.get("remediation", "No specific solution provided."))
            category = metadata.get("category", "N/A")
            root_cause = metadata.get("root_cause", "N/A")
            
            formatted_docs.append(
                f"--- Relevant Document {i+1} ---\n"
                f"Content: {content}\n"
                f"Category: {category}\n"
                f"Root Cause: {root_cause}\n"
                f"Solution: {solution}\n"
                f"---------------------------\n"
            )
        return "\n".join(formatted_docs)


# Example Usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting AirflowAgent example...")

    # Mock issue object for demonstration
    class MockIssue:
        def __init__(self, number, title, body):
            self.number = number
            self.title = title
            self.body = body

    # Mock log_data and enrichment_data
    mock_log_data_success = {
        "full_log": "ERROR - Task failed: ZeroDivisionError: division by zero in task 'transform_data'. Check data inputs.",
        "error_summary": "ZeroDivisionError in transform_data task.",
        "dag_id": "my_data_pipeline",
        "task_id": "transform_data",
        "dag_run_id": "manual__2023-01-01T00:00:00+00:00"
    }

    mock_enrichment_data_success = {
        "faiss_docs": [
            # Mock Document objects
            type('MockDoc', (object,), {
                'page_content': 'Task `transform_data` in `my_data_pipeline` failed due to ZeroDivisionError. This was fixed by adding input validation.',
                'metadata': {
                    'category': 'airflow', 
                    'root_cause': 'data_quality', 
                    'solution': 'Implement input validation for data processing steps.',
                    'remediation': 'Add validation code to transform_data task.'
                }
            })()
        ],
        "similar_issues_graph": [
            {"issueNumber": 99, "title": "DAG ABC failed - division by zero", "category": "airflow"},
            {"issueNumber": 98, "title": "Data quality issue in XYZ pipeline", "category": "airflow"}
        ],
        "related_neo4j_solutions": [
            {"solutionName": "Review data quality checks", "solutionDescription": "Examine if data quality checks are sufficient."}
        ]
    }
    
    # Mock log_data for a scenario where IDs are missing
    mock_log_data_missing_ids = {
        "full_log": "General Airflow error. Something went wrong.",
        "error_summary": "Generic error.",
        "dag_id": "", # Missing
        "task_id": "", # Missing
        "dag_run_id": ""
    }
    
    # Mock issue with no Airflow details
    mock_issue_no_airflow = MockIssue(
        number=103,
        title="General System Alert: Unknown error occurred",
        body="A general system error was observed in logs: Error XYZ. No specific DAG or task mentioned."
    )

    agent = AirflowAgent()

    print("\n--- Testing handle method with full context ---")
    mock_issue_1 = MockIssue(
        number=101,
        title="Airflow Alert: DAG `my_data_pipeline` - Task `transform_data` Failed (Run: manual__2023-01-01T00:00:00+00:00)",
        body=mock_log_data_success["full_log"]
    )
    handle_result_1 = agent.handle(mock_issue_1, mock_log_data_success, mock_enrichment_data_success)
    print(json.dumps(handle_result_1, indent=2))

    print("\n--- Testing handle method with missing Airflow IDs ---")
    mock_issue_2 = MockIssue(
        number=102,
        title="Airflow Alert: General Failure",
        body="Some log output indicates an issue, but no DAG or Task IDs were explicitly found."
    )
    handle_result_2 = agent.handle(mock_issue_2, mock_log_data_missing_ids, mock_enrichment_data_success)
    print(json.dumps(handle_result_2, indent=2))

    # Simulate execution (this would normally be triggered by webhook after approval)
    print("\n--- Simulating execute_remediation (assuming approval) ---")
    if handle_result_1.get("status") == "awaiting_approval":
        execute_result = agent.execute_remediation(
            issue_number=handle_result_1["issue_number"],
            dag_id=handle_result_1["dag_id"],
            task_id=handle_result_1["task_id"],
            dag_run_id=handle_result_1["dag_run_id"]
        )
        print(json.dumps(execute_result, indent=2))
    else:
        print("Skipping execute_remediation as the handle method did not return 'awaiting_approval'.")

    logger.info("AirflowAgent example finished.")
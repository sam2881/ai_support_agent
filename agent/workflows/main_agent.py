# agent/workflows/main_agent.py
import logging
import re
import json
from datetime import datetime # Import datetime for logging
from typing import Dict, Any, Tuple, Optional, List

from agent.clients.github_client import GitHubClient
from agent.clients.airflow_client import AirflowClient # Assuming this is your Airflow client
from agent.rag.vector_store import VectorStore, Document # Import Document for type hinting
from agent.neo4j.graph_client import GraphClient # Assuming this is your Neo4j client
from agent.models.llm_interface import LLMInterface # Import the LLMInterface

# Import sub-agents
# These sub-agents will now likely receive the LLMInterface instance
# to make their own specialized LLM calls for remediation.
from agent.workflows.airflow_agent import AirflowAgent
from agent.workflows.devops_agent import DevOpsAgent
from agent.workflows.access_agent import AccessAgent
# Removed: from agent.workflows.agents.router import determine_routing # No longer needed

from agent.workflows.approval_processor import ApprovalProcessor # Assuming you have this
from agent.workflows.config import settings # Assuming settings handles API keys and other configs

logger = logging.getLogger(__name__)

class MainAgent:
    def __init__(self):
        """
        Initializes the MainAgent with clients and sub-agents.
        All necessary external connections and sub-agents are instantiated here.
        """
        self.gh = GitHubClient()
        self.af = AirflowClient() # This should be your Airflow API client, not the old AirflowClient
        self.vector_store = VectorStore()
        self.graph = GraphClient()
        self.llm_interface = LLMInterface() # Initialize LLMInterface here
        self.approval_processor = ApprovalProcessor() # Initialize ApprovalProcessor

        # Initialize sub-agents. They will now get access to the LLMInterface and other clients.
        self.airflow_agent = AirflowAgent(
            llm_interface=self.llm_interface, 
            airflow_client=self.af, 
            github_client=self.gh,
            vector_store=self.vector_store, # Pass vector_store if sub-agents need to query it
            graph_client=self.graph # Pass graph_client if sub-agents need to interact with Neo4j
        )
        self.devops_agent = DevOpsAgent(
            llm_interface=self.llm_interface, 
            github_client=self.gh,
            vector_store=self.vector_store,
            graph_client=self.graph
        )
        self.access_agent = AccessAgent(
            llm_interface=self.llm_interface, 
            github_client=self.gh,
            vector_store=self.vector_store,
            graph_client=self.graph
        )

    def _parse_airflow_details_from_issue(self, issue_title: str, issue_body: str) -> Dict[str, str]:
        """
        Parses DAG ID, Task ID, and DAG Run ID from the GitHub issue title and body.
        These are crucial for fetching Airflow logs.

        Args:
            issue_title (str): The title of the GitHub issue.
            issue_body (str): The body of the GitHub issue.

        Returns:
            Dict[str, str]: A dictionary containing 'dag_id', 'task_id', 'dag_run_id'.
                            Returns empty strings if not found.
        """
        dag_id, task_id, dag_run_id = "", "", ""

        # Attempt to parse from title first (more structured)
        # Expected format: "🔥 Airflow Alert: DAG `dag_id` - Task `task_id` Failed (Run: dag_run_id)"
        title_match = re.search(r"DAG `([^`]+)` - Task `([^`]+)` Failed \(Run: ([^)]+)\)", issue_title)
        if title_match:
            dag_id = title_match.group(1)
            task_id = title_match.group(2)
            dag_run_id = title_match.group(3)
            logger.debug(f"Parsed from title: DAG={dag_id}, Task={task_id}, Run={dag_run_id}")
        else:
            # Fallback to parsing from body if title regex fails (e.g., if format changes)
            dag_id_match = re.search(r"DAG ID:\s*`([^`]+)`", issue_body)
            task_id_match = re.search(r"Task ID:\s*`([^`]+)`", issue_body)
            run_id_match = re.search(r"Run ID:\s*`([^`]+)`", issue_body)

            if dag_id_match: dag_id = dag_id_match.group(1)
            if task_id_match: task_id = task_id_match.group(1)
            if run_id_match: dag_run_id = run_id_match.group(1)
            
            if dag_id or task_id or dag_run_id:
                logger.debug(f"Parsed from body: DAG={dag_id}, Task={task_id}, Run={dag_run_id}")

        return {"dag_id": dag_id, "task_id": task_id, "dag_run_id": dag_run_id}

    def enrich_issue(self, issue_text: str, full_log: str) -> Dict[str, Any]:
        """
        Enriches the issue with relevant historical data from RAG sources (Vector Store, Neo4j).
        This data will be passed to the LLM for context.

        Args:
            issue_text (str): Combined issue title and body.
            full_log (str): The full log content from Airflow.

        Returns:
            Dict[str, Any]: A dictionary containing relevant 'categories', 'root_causes', 'solutions',
                            and 'similar_issues' found in the RAG sources, and the raw FAISS documents.
        """
        combined_query_text = issue_text + "\n\n" + full_log[:4000] # Limit log for RAG query if too long

        # Query Vector Store for historical documents
        top_docs: List[Document] = self.vector_store.query(combined_query_text, k=4)
        
        # Extract metadata from top documents (for potential structured use, though LLM handles raw docs well)
        categories_from_docs = [d.metadata.get("category", "").lower() for d in top_docs if d.metadata.get("category")]
        root_causes_from_docs = [d.metadata.get("root_cause", "") for d in top_docs if d.metadata.get("root_cause")]
        solutions_from_docs = [d.metadata.get("solution", "") for d in top_docs if d.metadata.get("solution")]

        # Query Neo4j for similar issues (based on keyword search in issue title/body)
        similar_issues_from_graph = self.graph.query_similar_issues(combined_query_text, limit=3)
        
        # Optionally, get related categories/solutions directly from Neo4j based on initial keyword matches
        related_neo4j_solutions = []
        if "airflow" in issue_text.lower(): # Simple keyword check
            related_neo4j_solutions.extend(self.graph.get_related_solutions("airflow"))
        if "devops" in issue_text.lower() or "kubernetes" in issue_text.lower():
            related_neo4j_solutions.extend(self.graph.get_related_solutions("devops"))


        return {
            "faiss_docs": top_docs, # Pass full Document objects for easier handling in LLM context creation
            "faiss_categories": list(set(categories_from_docs)),
            "faiss_root_causes": list(set(root_causes_from_docs)),
            "faiss_solutions": list(set(solutions_from_docs)),
            "similar_issues_graph": similar_issues_from_graph,
            "related_neo4j_solutions": related_neo4j_solutions
        }

    def log_routing_event(self, issue_number: int, category: str, routing_reason: str):
        """
        Logs the routing decision to a local audit file and Neo4j.
        """
        timestamp = datetime.now().isoformat()
        description = (
            f"[{timestamp}] Issue #{issue_number} auto-classified as '{category}'. "
            f"Reason: {routing_reason}"
        )
        try:
            # Ensure the 'data' directory exists for persistent logs
            log_dir = settings.DATA_DIR # Assuming DATA_DIR from settings
            os.makedirs(log_dir, exist_ok=True)
            log_filepath = os.path.join(log_dir, "routing_audit.log")
            with open(log_filepath, "a") as log:
                log.write(description + "\n")
            logger.info(f"Logged routing event to file for issue #{issue_number}.")
        except IOError as e:
            logger.error(f"Failed to write to routing_audit.log: {e}")

        # Log to Neo4j graph
        self.graph.log_issue_classification(issue_number, category, description)
        self.graph.log_agent_relationship(issue_number, category, status="classified")
        logger.info(f"Logged routing event to Neo4j for issue #{issue_number}.")

    def delegate_to_agent(self, category: str, issue: Any, log_data: Dict[str, str], enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegates the issue to the appropriate sub-agent based on the classification.

        Args:
            category (str): The classified category of the issue (e.g., "airflow", "devops").
            issue (Any): The GitHub issue object.
            log_data (Dict[str, str]): The extracted log data.
            enrichment_data (Dict[str, Any]): The RAG enrichment data.

        Returns:
            Dict[str, Any]: The result from the delegated agent's handling method.
        """
        logger.info(f"Delegating issue #{issue.number} to {category.upper()} Agent...")
        if category == "airflow":
            return self.airflow_agent.handle(issue, log_data, enrichment_data)
        elif category == "devops":
            return self.devops_agent.handle(issue, log_data, enrichment_data)
        elif category == "access":
            return self.access_agent.handle(issue, log_data, enrichment_data)
        else:
            comment = "⚠️ Main Agent: Unable to classify issue definitively. Please assign manually."
            self.gh.add_comment(issue.number, comment)
            logger.warning(f"⚠️ Issue #{issue.number} could not be classified or routed to a known agent.")
            return {"status": "unrouted", "llm_summary": comment, "auto_approved": False}

    def process_issue(self, issue_number: int) -> Dict[str, Any]:
        """
        Main entry point for processing a GitHub issue.
        It fetches issue details, extracts relevant data, enriches it,
        classifies the issue, and delegates to the appropriate sub-agent.

        Args:
            issue_number (int): The number of the GitHub issue to process.

        Returns:
            Dict[str, Any]: A summary of the processing outcome.
        """
        logger.info(f"⚡ Starting processing for GitHub issue #{issue_number}")

        # 1. Fetch GitHub Issue
        issue = self.gh.get_issue(issue_number)
        if not issue:
            logger.error(f"❌ Could not retrieve GitHub issue #{issue_number}. Aborting processing.")
            return {"status": "failed", "message": f"Issue {issue_number} not found or accessible."}
        
        issue_text = f"{issue.title}\n{issue.body or ''}"
        
        # 2. Extract Airflow Details from Issue and Fetch Logs
        airflow_meta = self._parse_airflow_details_from_issue(issue.title, issue.body)
        log_data = {
            "full_log": "", "error_summary": "", "traceback": "", 
            "container_id": "", "log_tail": "",
            **airflow_meta # Add parsed dag_id, task_id, dag_run_id to log_data for sub-agents
        }

        if airflow_meta["dag_id"] and airflow_meta["task_id"] and airflow_meta["dag_run_id"]:
            try:
                # Fetch full logs from Airflow
                fetched_log_data = self.af.get_task_logs(
                    airflow_meta["dag_id"], 
                    airflow_meta["dag_run_id"], 
                    airflow_meta["task_id"]
                )
                log_data.update(fetched_log_data)
                logger.info(f"✅ Fetched Airflow logs for DAG: {airflow_meta['dag_id']}, Task: {airflow_meta['task_id']}")
            except Exception as e:
                logger.warning(f"❌ Failed to fetch full Airflow logs for issue #{issue_number}: {e}")
                self.gh.add_comment(issue_number, f"⚠️ Automated system failed to fetch full Airflow logs (Error: {e}). Proceeding with available log excerpt.")
        else:
            logger.info(f"No complete Airflow DAG/Task/Run IDs found in issue #{issue_number} title/body. Skipping full log fetch.")
            self.gh.add_comment(issue_number, "ℹ️ Automated system could not parse DAG, Task, or Run IDs from issue. Cannot fetch full Airflow logs.")
            # Use the log_excerpt from the issue body as a fallback for initial classification
            body_log_excerpt_match = re.search(r"Recent Log Output \(Last \d+ lines of excerpt\):\n```log\n([\s\S]+?)\n```", issue.body)
            if body_log_excerpt_match:
                log_data["full_log"] = body_log_excerpt_match.group(1)
                log_data["log_tail"] = log_data["full_log"] # For simplicity, treat as tail
                log_data["error_summary"] = self.gh.extract_error_message(log_data["full_log"]) # Re-use GitHubClient's error extraction
                logger.info(f"Using log excerpt from issue body for analysis.")

        # 3. Enrich Issue with RAG (Retrieval Augmented Generation) Context
        enrichment_data = self.enrich_issue(issue_text, log_data["full_log"])
        logger.info(f"📚 Enriched issue #{issue_number} with RAG context. Found {len(enrichment_data['faiss_docs'])} FAISS docs and {len(enrichment_data['similar_issues_graph'])} Neo4j issues.")

        # Prepare context for LLM (combine FAISS docs and Neo4j info into a single string)
        # The LLMInterface expects a combined string context for get_routing_suggestion
        context_for_llm = ""
        if enrichment_data["faiss_docs"]:
            context_for_llm += "Relevant historical issues from Knowledge Base:\n"
            for doc in enrichment_data["faiss_docs"]:
                context_for_llm += f"- Title: {doc.metadata.get('title', 'N/A')}\n  Summary: {doc.page_content}\n  Category: {doc.metadata.get('category', 'N/A')}\n  Solution: {doc.metadata.get('solution', 'N/A')}\n"
        
        if enrichment_data["similar_issues_graph"]:
            context_for_llm += "\nSimilar issues from Graph Database:\n"
            for similar_issue in enrichment_data["similar_issues_graph"]:
                context_for_llm += f"- Issue ID: {similar_issue.get('issue_id', 'N/A')}, Summary: {similar_issue.get('summary', 'N/A')}\n"
        
        if enrichment_data["related_neo4j_solutions"]:
            context_for_llm += "\nRelated solutions from Graph Database:\n"
            for solution in enrichment_data["related_neo4j_solutions"]:
                context_for_llm += f"- {solution}\n"

        # 4. Classify/Route the Issue using the LLMInterface
        # The LLMInterface handles the prompt construction and LLM call
        routing_result = self.llm_interface.get_routing_suggestion(
            incident_description=issue_text + "\n" + log_data["full_log"], # Pass relevant info to LLM
            # Note: get_routing_suggestion in LLMInterface does not currently take `context` directly
            # It builds its own context from system prompt and few-shot examples.
            # If you want to use the RAG context here, you'd need to modify get_routing_suggestion
            # or prepend it to the incident_description. For now, LLM uses its internal prompt for routing.
        )
        category = routing_result.get("agent", "general")
        routing_reason = routing_result.get("routing_reason", "LLM provided no specific reason.")
        
        logger.info(f"🔀 Issue #{issue_number} classified as '{category}'. Reason: {routing_reason}")
        self.log_routing_event(issue_number, category, routing_reason)

        # 5. Delegate to Sub-Agent
        # Sub-agents now receive the LLMInterface and other clients during their initialization.
        # They will make their own LLM calls for remediation.
        result = self.delegate_to_agent(category, issue, log_data, enrichment_data)
        
        # 6. Final Status Update / Comment
        llm_summary = result.get("llm_summary", "No summary from agent.")
        action_payload = result.get("action_payload", {}) # Extract action_payload

        # Add to approval queue if an action_payload exists and is not already auto-approved
        if action_payload and not result.get("auto_approved", False):
            self.approval_processor.add_to_queue(
                issue_number=issue_number,
                agent=category,
                summary=llm_summary,
                payload=action_payload
            )
            comment_status = "awaiting approval"
            self.gh.add_comment(issue_number, f"ℹ️ Main Agent: Issue processed. Remediation suggested by {category.upper()} Agent and added to approval queue. Summary: {llm_summary}")
        else:
            comment_status = result.get("status", "unknown")
            if result.get("status") == "resolved" or result.get("status") == "auto_cleared":
                self.gh.close_issue(issue_number, f"✅ Issue resolved and closed by {category.upper()} Agent. {llm_summary}")
                self.graph.log_agent_relationship(issue_number, category, status="resolved")
                logger.info(f"Issue #{issue_number} closed.")
            elif result.get("status") == "processed":
                self.gh.add_comment(issue_number, f"ℹ️ Main Agent: Issue processed and delegated. Summary: {llm_summary}")
                self.graph.log_agent_relationship(issue_number, category, status="processed")
            elif result.get("status") == "unrouted": # Handled in delegate_to_agent already
                 pass
            else:
                 self.gh.add_comment(issue_number, f"❓ Main Agent: Issue processed with unknown status from {category.upper()} Agent. Summary: {llm_summary}")

        final_summary = {
            "issue_number": issue_number,
            "agent": category,
            "status": comment_status, # Reflect whether it's awaiting approval
            "auto_approved": result.get("auto_approved", False),
            "llm_summary": llm_summary,
            "action_payload": action_payload, # Include for potential UI display
            "routing_reason": routing_reason,
            "enrichment_details": enrichment_data
        }
        logger.info(f"✨ Finished processing issue #{issue_number}. Status: {final_summary['status']}")
        return final_summary

# Example Usage (for testing purposes)
if __name__ == '__main__':
    from dotenv import load_dotenv
    from unittest.mock import patch # For mocking settings and other dependencies for isolated testing
    import sys
    import os

    # Add agent directory to sys.path if running this script directly
    # This helps resolve relative imports like agent.models.llm_interface
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    load_dotenv(os.path.join(project_root, '.env')) # Load .env from project root

    logging.basicConfig(
        level=logging.DEBUG, # Use DEBUG for more verbosity during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting MainAgent example...")

    # Mock necessary dependencies for local testing without full Docker Compose setup
    # In a real deployed environment, these would be real instances.
    class MockGitHubClient:
        def get_issue(self, issue_number: int):
            if issue_number == 1:
                return type('obj', (object,), {
                    'number': 1,
                    'title': '🔥 Airflow Alert: DAG `data_pipeline` - Task `transform_data` Failed (Run: 2024-05-22T10:00:00+00:00)',
                    'body': 'Task `transform_data` failed with exit code 1. Logs:\n```log\n[2024-05-22 10:05:00] ERROR - Task failed with exit code 1. Database connection timed out. Could not reach `db-replica-1`.\n```'
                })()
            elif issue_number == 2:
                return type('obj', (object,), {
                    'number': 2,
                    'title': 'Kubernetes Pod CrashLoopBackOff: `api-gateway` due to OOMKilled',
                    'body': 'The `api-gateway` pod is repeatedly restarting in `CrashLoopBackOff` state. Logs show `OOMKilled`. Needs more memory allocation.\n```log\n[2024-05-22 11:30:00] INFO - Starting api-gateway...\n[2024-05-22 11:30:05] ERROR - OutOfMemory: Failed to allocate 1GB\n```'
                })()
            elif issue_number == 3:
                return type('obj', (object,), {
                    'number': 3,
                    'title': 'User permissions issue: John Doe cannot access Production S3 bucket',
                    'body': 'User john.doe reports "Access Denied" when trying to list objects in `s3://prod-data-lake/`. Please review IAM policies.'
                })()
            return None
        def add_comment(self, issue_number: int, comment: str):
            print(f"\n--- MOCK GitHub Comment on Issue #{issue_number} ---\n{comment}\n------------------------------------------")
        def close_issue(self, issue_number: int, comment: str):
            print(f"\n--- MOCK GitHub Closing Issue #{issue_number} ---\n{comment}\n------------------------------------------")
        def extract_error_message(self, log_content: str) -> str:
            match = re.search(r"ERROR - (.+)", log_content)
            return match.group(1) if match else "Error message not found."

    class MockAirflowClient:
        def get_task_logs(self, dag_id: str, dag_run_id: str, task_id: str) -> Dict[str, str]:
            print(f"MOCK: Fetching logs for DAG: {dag_id}, Task: {task_id}, Run: {dag_run_id}")
            if dag_id == "data_pipeline" and task_id == "transform_data":
                return {"full_log": "[2024-05-22 10:05:00] INFO - Starting data transformation...\n[2024-05-22 10:06:00] ERROR - Database connection timed out. Could not reach `db-replica-1`. Retrying in 5s.\n[2024-05-22 10:07:00] ERROR - Max retries exceeded. Task failed.\n", "error_summary": "Database connection timed out."}
            return {"full_log": "No specific log found for this task.", "error_summary": "N/A"}

    class MockVectorStore:
        def query(self, query_text: str, k: int) -> List[Document]:
            print(f"MOCK: Querying VectorStore for: {query_text[:50]}...")
            # Simulate relevant documents
            if "database connection" in query_text.lower():
                return [
                    Document(page_content="Common issue: database connection timeouts due to replica overload.", metadata={"category": "airflow", "root_cause": "database overload", "solution": "check DB metrics, scale replica, or retry task."}),
                    Document(page_content="Airflow task 'db_cleanup' failed. Root cause: incorrect database credentials.", metadata={"category": "airflow", "root_cause": "db credentials", "solution": "update connection in Airflow UI."})
                ]
            elif "oomkilled" in query_text.lower() or "memory" in query_text.lower():
                return [
                    Document(page_content="Kubernetes pod OOMKilled. Solution: Increase resource limits in deployment YAML.", metadata={"category": "devops", "root_cause": "insufficient memory", "solution": "kubectl apply -f deployment.yaml with higher limits."})
                ]
            elif "permission denied" in query_text.lower() or "access denied" in query_text.lower():
                return [
                    Document(page_content="S3 Access Denied for user. Solution: Verify IAM policy attachments for the user/role.", metadata={"category": "access", "root_cause": "incorrect IAM policy", "solution": "Review IAM policies."})
                ]
            return []
        # Ensure query method for VectorStore returns Document objects as expected

    class MockGraphClient:
        def query_similar_issues(self, query: str, limit: int) -> List[Dict[str, Any]]:
            print(f"MOCK: Querying Neo4j for similar issues: {query[:50]}...")
            if "database" in query.lower():
                return [{"issue_id": 101, "summary": "DB timeout on data_load DAG"}]
            elif "kubernetes" in query.lower():
                return [{"issue_id": 205, "summary": "K8s pod restart loop OOM"}]
            return []
        def get_related_solutions(self, category: str) -> List[str]:
            print(f"MOCK: Getting related Neo4j solutions for: {category}")
            if category == "airflow": return ["Check Airflow DB health", "Restart Airflow scheduler"]
            if category == "devops": return ["Adjust K8s resource requests", "Check container logs"]
            return []
        def log_issue_classification(self, *args, **kwargs):
            print(f"MOCK: Logging issue classification to Neo4j.")
        def log_agent_relationship(self, *args, **kwargs):
            print(f"MOCK: Logging agent relationship to Neo4j.")

    # Mock the LLMInterface and the specific sub-agents
    class MockLLMInterface(LLMInterface):
        def __init__(self, model_name: Optional[str] = None):
            # No need to call super().__init__, just define mock behavior
            pass 
        def get_routing_suggestion(self, incident_description: str) -> Dict[str, str]:
            print(f"MOCK LLM: Getting routing suggestion for: {incident_description[:50]}...")
            if "airflow" in incident_description.lower() or "dag" in incident_description.lower():
                return {"agent": "airflow", "routing_reason": "Identified Airflow DAG/task failure keywords."}
            elif "kubernetes" in incident_description.lower() or "pod" in incident_description.lower():
                return {"agent": "devops", "routing_reason": "Detected Kubernetes infrastructure terms."}
            elif "permission denied" in incident_description.lower() or "access" in incident_description.lower():
                return {"agent": "access", "routing_reason": "Recognized access control related phrases."}
            return {"agent": "general", "routing_reason": "No specific agent keywords found."}

        def get_remediation_suggestion(self, agent_type: str, incident_details: str, context: Optional[str] = None) -> Dict[str, Any]:
            print(f"MOCK LLM: Getting remediation for {agent_type}: {incident_details[:50]}... (Context: {context[:50] if context else 'N/A'})")
            if agent_type == "airflow":
                return {"llm_summary": "MOCK: Airflow remediation: Consider re-triggering the DAG or checking database connectivity.", "action_payload": {"type": "airflow_retrigger", "dag_id": "mock_dag", "task_id": "mock_task", "run_id": "mock_run"}}
            elif agent_type == "devops":
                return {"llm_summary": "MOCK: DevOps remediation: Increase Kubernetes pod memory limits in deployment YAML.", "action_payload": {"type": "kubernetes_scale", "resource": "memory", "target_pod": "mock_pod"}}
            elif agent_type == "access":
                return {"llm_summary": "MOCK: Access remediation: Review IAM policies for user 'john.doe' and ensure S3 bucket permissions are correct.", "action_payload": {"type": "iam_review", "user": "john.doe"}}
            return {"llm_summary": "MOCK: General remediation: Please manually investigate the issue further.", "auto_approved": False}

    class MockAirflowAgent:
        def __init__(self, **kwargs):
            self.llm_interface = kwargs.get('llm_interface')
            self.github_client = kwargs.get('github_client')
            print("MOCK: AirflowAgent initialized.")
        def handle(self, issue: Any, log_data: Dict[str, str], enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
            print(f"MOCK: AirflowAgent handling issue #{issue.number}...")
            # Simulate generating remediation
            remediation_result = self.llm_interface.get_remediation_suggestion(
                agent_type="airflow",
                incident_details=log_data.get("full_log", ""),
                context="\n".join([d.page_content for d in enrichment_data["faiss_docs"]])
            )
            return {"status": "processed", "llm_summary": remediation_result["llm_summary"], "action_payload": remediation_result.get("action_payload")}

    class MockDevOpsAgent:
        def __init__(self, **kwargs):
            self.llm_interface = kwargs.get('llm_interface')
            self.github_client = kwargs.get('github_client')
            print("MOCK: DevOpsAgent initialized.")
        def handle(self, issue: Any, log_data: Dict[str, str], enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
            print(f"MOCK: DevOpsAgent handling issue #{issue.number}...")
            remediation_result = self.llm_interface.get_remediation_suggestion(
                agent_type="devops",
                incident_details=issue.body,
                context="\n".join([d.page_content for d in enrichment_data["faiss_docs"]])
            )
            return {"status": "processed", "llm_summary": remediation_result["llm_summary"], "action_payload": remediation_result.get("action_payload")}
            
    class MockAccessAgent:
        def __init__(self, **kwargs):
            self.llm_interface = kwargs.get('llm_interface')
            self.github_client = kwargs.get('github_client')
            print("MOCK: AccessAgent initialized.")
        def handle(self, issue: Any, log_data: Dict[str, str], enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
            print(f"MOCK: AccessAgent handling issue #{issue.number}...")
            remediation_result = self.llm_interface.get_remediation_suggestion(
                agent_type="access",
                incident_details=issue.body,
                context="\n".join([d.page_content for d in enrichment_data["faiss_docs"]])
            )
            return {"status": "processed", "llm_summary": remediation_result["llm_summary"], "action_payload": remediation_result.get("action_payload")}

    class MockApprovalProcessor:
        def __init__(self):
            print("MOCK: ApprovalProcessor initialized.")
        def add_to_queue(self, issue_number: int, agent: str, summary: str, payload: Dict[str, Any]):
            print(f"MOCK: Added to approval queue: Issue #{issue_number}, Agent: {agent}, Summary: {summary[:50]}..., Payload: {payload}")


    # Temporarily replace real clients with mocks for testing
    with patch('agent.workflows.main_agent.GitHubClient', MockGitHubClient), \
         patch('agent.workflows.main_agent.AirflowClient', MockAirflowClient), \
         patch('agent.workflows.main_agent.VectorStore', MockVectorStore), \
         patch('agent.workflows.main_agent.GraphClient', MockGraphClient), \
         patch('agent.workflows.main_agent.LLMInterface', MockLLMInterface), \
         patch('agent.workflows.main_agent.AirflowAgent', MockAirflowAgent), \
         patch('agent.workflows.main_agent.DevOpsAgent', MockDevOpsAgent), \
         patch('agent.workflows.main_agent.AccessAgent', MockAccessAgent), \
         patch('agent.workflows.main_agent.ApprovalProcessor', MockApprovalProcessor):
        
        # Mock settings as well, especially for DATA_DIR
        class MockSettings:
            DATA_DIR = os.path.join(project_root, 'data') # Ensure this points to a writable path
            # Include any other settings that main_agent or its dependencies might need
            DEFAULT_LLM_MODEL = "mock-llm" # Just for LLMInterface init, not used by MockLLMInterface
            OPENAI_API_KEY = "mock_key"
            GEMINI_API_KEY = "mock_key"
            # ... add any other settings needed by dependencies

        with patch('agent.workflows.main_agent.settings', MockSettings()):
            main_agent = MainAgent()

            # --- Simulate a new GitHub issue coming in ---
            test_issues = [1, 2, 3] # Test with Airflow, DevOps, Access examples

            for issue_num in test_issues:
                print(f"\n--- Processing Issue #{issue_num} ---")
                try:
                    processing_result = main_agent.process_issue(issue_num)
                    print(json.dumps(processing_result, indent=2))
                except Exception as e:
                    logger.error(f"Error processing test issue #{issue_num}: {e}", exc_info=True)
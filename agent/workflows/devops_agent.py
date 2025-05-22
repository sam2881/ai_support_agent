# agent/workflows/agents/devops_agent.py
import logging
import json
from datetime import datetime
from typing import Dict, Any, List

from agent.clients.github_client import GitHubClient
from agent.workflows.config import settings
from agent.utils.queue_utils import ApprovalQueue
from agent.models.prompts import DEVOPS_SOLUTION_PROMPT # New prompt specifically for DevOps

# Import the correct LLM library based on settings.LLM_MODEL
if "gemini" in settings.LLM_MODEL:
    import google.generativeai as genai
    genai.configure(api_key=settings.GOOGLE_API_KEY.get_secret_value())
    LLM_CLIENT = genai.GenerativeModel(settings.LLM_MODEL)
    logger.info(f"Initialized Google Generative AI client with model: {settings.LLM_MODEL}")
else:
    from langchain_openai import ChatOpenAI
    LLM_CLIENT = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=0,
        openai_api_key=settings.OPENAI_API_KEY.get_secret_value()
    )
    logger.info(f"Initialized Langchain ChatOpenAI client with model: {settings.LLM_MODEL}")

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class DevOpsAgent:
    def __init__(self):
        """
        Initializes the DevOpsAgent with necessary clients and utilities.
        """
        self.gh = GitHubClient()
        self.queue = ApprovalQueue()

    def handle(self, issue: Any, log_data: Dict[str, str], enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles a DevOps-related GitHub issue by analyzing logs and RAG context,
        proposing a remediation, and submitting it for approval.

        Args:
            issue (Any): The GitHub issue object (from PyGithub).
            log_data (Dict[str, str]): Relevant log data.
            enrichment_data (Dict[str, Any]): Context from RAG sources (VectorStore, Neo4j)
                                              passed by the MainAgent.

        Returns:
            Dict[str, Any]: A dictionary containing the status of the agent's operation,
                            including proposed actions and LLM summary.
        """
        issue_number = issue.number
        issue_text = f"{issue.title}\n{issue.body or ''}"
        full_log = log_data.get("full_log", "") # Use full_log for comprehensive analysis

        try:
            # Prepare RAG context for the LLM prompt
            faiss_context = self._format_rag_docs(enrichment_data.get("faiss_docs", []))
            graph_similar_issues = json.dumps(enrichment_data.get("similar_issues_graph", []), indent=2)
            graph_related_solutions = json.dumps(enrichment_data.get("related_neo4j_solutions", []), indent=2)

            # Construct the LLM prompt using the DEVOPS_SOLUTION_PROMPT template
            prompt_content = DEVOPS_SOLUTION_PROMPT.format(
                issue_title=issue.title,
                issue_body=issue.body,
                full_log=full_log,
                faiss_context=faiss_context,
                graph_similar_issues=graph_similar_issues,
                graph_related_solutions=graph_related_solutions
            )

            # Call LLM and parse the response
            parser = JsonOutputParser()
            
            if "gemini" in settings.LLM_MODEL:
                response = LLM_CLIENT.generate_content(prompt_content)
                llm_response_content = response.candidates[0].content.parts[0].text
            else:
                messages = [
                    SystemMessage(content="You are a helpful DevOps assistant that provides solutions in JSON format."),
                    HumanMessage(content=prompt_content),
                ]
                llm_response_content = LLM_CLIENT.invoke(messages).content

            parsed = parser.parse(llm_response_content)

            proposed_action = parsed.get("proposed_action", "No specific action proposed.")
            reason = parsed.get("reason", "No reason provided by LLM.")
            confidence_score = parsed.get("confidence_score", 0.0)
            target_component = parsed.get("target_component", "Unknown Component")
            
            summary = (
                f"🛠️ **DevOps Agent Proposed Remediation**:\n\n"
                f"**Proposed Action**: {proposed_action}\n"
                f"**Target Component**: {target_component}\n"
                f"**Reasoning**: {reason}\n"
                f"**Confidence Score**: {confidence_score:.2f} (0.0-1.0)\n\n"
                f"Waiting for manual approval to proceed with execution."
            )
            
            self.gh.add_comment(issue_number, summary)
            logger.info(f"Issue #{issue_number}: DevOps proposed remediation posted.")

            # Submit for approval
            queue_key = f"devops-{issue_number}-{target_component}" # Unique key for DevOps actions
            if not self.queue.exists(issue_number, queue_key):
                retry_count = self.queue.retry_count(issue_number, queue_key)
                self.queue.submit(
                    agent="devops",
                    issue_number=issue_number,
                    payload={
                        "target_component": target_component,
                        "proposed_action": proposed_action,
                        "reason": reason,
                        "confidence_score": confidence_score,
                        "retry_count": retry_count,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    },
                    summary=summary
                )
                logger.info(f"Issue #{issue_number}: DevOps action submitted to approval queue.")
            else:
                logger.info(f"Issue #{issue_number}: DevOps action already in approval queue. Skipping re-submission.")

            return {
                "agent": "devops",
                "status": "awaiting_approval",
                "target_component": target_component,
                "llm_summary": summary,
                "auto_approved": False
            }

        except json.JSONDecodeError as e:
            error_msg = f"LLM response was not valid JSON. Error: {e}\nRaw response: {llm_response_content[:500]}..."
            logger.error(f"❌ DevOpsAgent JSON parsing failed for issue #{issue_number}: {error_msg}", exc_info=True)
            self.gh.add_comment(issue_number, f"⚠️ DevOps Agent: Failed to parse LLM response. {error_msg}")
            return {"agent": "devops", "status": "failed", "error": error_msg}
        except Exception as e:
            error_msg = f"DevOpsAgent encountered an unexpected error: {e}"
            logger.error(f"❌ {error_msg} for issue #{issue_number}", exc_info=True)
            self.gh.add_comment(issue_number, f"⚠️ DevOps Agent: Remediation process failed. Error: `{e}`")
            return {"agent": "devops", "status": "failed", "error": str(e)}

    def execute_remediation(self, issue_number: int, target_component: str, proposed_action: str) -> Dict[str, Any]:
        """
        Simulates the execution of a proposed DevOps remediation.
        In a real-world scenario, this would trigger playbooks,
        run scripts (e.g., Ansible, Terraform), or interact with cloud APIs.

        Args:
            issue_number (int): The GitHub issue number.
            target_component (str): The name of the component to target (e.g., 'Kubernetes cluster', 'EC2 instance').
            proposed_action (str): The specific action to take (e.g., 'restart pod', 'scale up instance').

        Returns:
            Dict[str, Any]: Status of the execution.
        """
        logger.info(f"Attempting to execute DevOps remediation for Issue #{issue_number}: Action '{proposed_action}' on '{target_component}'.")
        msg = ""
        try:
            # Placeholder for actual DevOps automation.
            # In a real system, you'd call specific DevOps tools/APIs here.
            # For example:
            # if "restart pod" in proposed_action.lower() and "kubernetes" in target_component.lower():
            #    k8s_client.restart_pod(target_component_name)
            # elif "resize vm" in proposed_action.lower():
            #    aws_client.resize_ec2(target_component_name)
            
            # Simulate success for demonstration
            msg = (
                f"✅ **DevOps Action Executed**:\n"
                f"Simulated execution of '{proposed_action}' on '{target_component}'.\n"
                f"In a real system, this would trigger an automated playbook or script."
            )
            logger.info(f"Successfully simulated DevOps action for issue #{issue_number}.")
            
            return {"status": "resolved", "message": msg, "auto_approved": True}
        except Exception as e:
            msg = f"❌ **DevOps Remediation Failed**:\n" \
                  f"Could not execute '{proposed_action}' on '{target_component}'.\n" \
                  f"Error: `{e}`\n" \
                  f"Manual intervention required."
            logger.error(f"Failed to execute DevOps action for issue #{issue_number}: {e}", exc_info=True)
            self.gh.add_comment(issue_number, msg)
            return {"status": "failed", "message": msg, "auto_approved": False}

    def _format_rag_docs(self, docs: List[Any]) -> str:
        """
        Helper to format RAG documents into a string for the LLM prompt.
        """
        if not docs:
            return "No relevant historical information found."

        formatted_docs = []
        for i, doc in enumerate(docs):
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

# Example Usage
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting DevOpsAgent example...")

    class MockIssue:
        def __init__(self, number, title, body):
            self.number = number
            self.title = title
            self.body = body

    mock_log_data = {
        "full_log": "ERROR - Kubernetes pod 'my-app-xyz' is in CrashLoopBackOff state. Image pull failed: rpc error: code = NotFound",
        "error_summary": "Image pull failed in Kubernetes pod.",
        "container_id": "xyz123"
    }

    mock_enrichment_data = {
        "faiss_docs": [
            type('MockDoc', (object,), {
                'page_content': 'Kubernetes pod in CrashLoopBackOff due to ImagePullBackOff. Check image registry access and image name.',
                'metadata': {'category': 'devops', 'root_cause': 'image_pull_failure', 'solution': 'Verify image path and registry credentials.'}
            })()
        ],
        "similar_issues_graph": [
            {"issueNumber": 201, "title": "K8s pod stuck in pending", "category": "devops"}
        ],
        "related_neo4j_solutions": [
            {"solutionName": "Check container registry access", "solutionDescription": "Ensure Kubernetes nodes have access to container registry."}
        ]
    }

    devops_agent = DevOpsAgent()

    print("\n--- Testing handle method for DevOps issue ---")
    mock_issue_devops = MockIssue(
        number=202,
        title="Urgent: Kubernetes Pod Failing - CrashLoopBackOff",
        body="Detailed logs here indicating a container image issue..."
    )
    handle_result_devops = devops_agent.handle(mock_issue_devops, mock_log_data, mock_enrichment_data)
    print(json.dumps(handle_result_devops, indent=2))

    print("\n--- Simulating execute_remediation for DevOps issue ---")
    if handle_result_devops.get("status") == "awaiting_approval":
        execute_result_devops = devops_agent.execute_remediation(
            issue_number=handle_result_devops["issue_number"],
            target_component=handle_result_devops.get("target_component", "K8s Pod"),
            proposed_action=handle_result_devops.get("llm_summary", "").split('\n')[2].replace('**Proposed Action**: ', '') # Extract for example
        )
        print(json.dumps(execute_result_devops, indent=2))

    logger.info("DevOpsAgent example finished.")
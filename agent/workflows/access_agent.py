# agent/workflows/agents/access_agent.py
import logging
import json
from datetime import datetime
from typing import Dict, Any, List

from agent.clients.github_client import GitHubClient
from agent.workflows.config import settings
from agent.utils.queue_utils import ApprovalQueue
from agent.models.prompts import ACCESS_SOLUTION_PROMPT # New prompt specifically for Access issues

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

class AccessAgent:
    def __init__(self):
        """
        Initializes the AccessAgent with necessary clients and utilities.
        """
        self.gh = GitHubClient()
        self.queue = ApprovalQueue()

    def handle(self, issue: Any, log_data: Dict[str, str], enrichment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles an Access-related GitHub issue by analyzing issue description and RAG context,
        proposing an access remediation, and submitting it for approval.

        Args:
            issue (Any): The GitHub issue object (from PyGithub).
            log_data (Dict[str, str]): Relevant log data (might be empty or less relevant for access issues).
            enrichment_data (Dict[str, Any]): Context from RAG sources (VectorStore, Neo4j)
                                              passed by the MainAgent.

        Returns:
            Dict[str, Any]: A dictionary containing the status of the agent's operation,
                            including proposed actions and LLM summary.
        """
        issue_number = issue.number
        issue_text = f"{issue.title}\n{issue.body or ''}"
        full_log = log_data.get("full_log", "") # May be less critical for access issues

        try:
            # Prepare RAG context for the LLM prompt
            faiss_context = self._format_rag_docs(enrichment_data.get("faiss_docs", []))
            graph_similar_issues = json.dumps(enrichment_data.get("similar_issues_graph", []), indent=2)
            graph_related_solutions = json.dumps(enrichment_data.get("related_neo4j_solutions", []), indent=2)

            # Construct the LLM prompt using the ACCESS_SOLUTION_PROMPT template
            prompt_content = ACCESS_SOLUTION_PROMPT.format(
                issue_title=issue.title,
                issue_body=issue.body,
                full_log=full_log, # Include logs even if less relevant for some access issues
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
                    SystemMessage(content="You are a helpful Access Management assistant that provides solutions in JSON format."),
                    HumanMessage(content=prompt_content),
                ]
                llm_response_content = LLM_CLIENT.invoke(messages).content

            parsed = parser.parse(llm_response_content)

            proposed_action = parsed.get("proposed_action", "No specific access action proposed.")
            target_user_group = parsed.get("target_user_group", "Unknown User/Group")
            resource = parsed.get("resource", "Unknown Resource")
            reason = parsed.get("reason", "No reason provided by LLM.")
            confidence_score = parsed.get("confidence_score", 0.0)
            
            summary = (
                f"🔑 **Access Agent Proposed Remediation**:\n\n"
                f"**Proposed Action**: {proposed_action}\n"
                f"**Target User/Group**: {target_user_group}\n"
                f"**Resource**: {resource}\n"
                f"**Reasoning**: {reason}\n"
                f"**Confidence Score**: {confidence_score:.2f} (0.0-1.0)\n\n"
                f"Waiting for manual approval to proceed with execution."
            )
            
            self.gh.add_comment(issue_number, summary)
            logger.info(f"Issue #{issue_number}: Access proposed remediation posted.")

            # Submit for approval
            queue_key = f"access-{issue_number}-{target_user_group}-{resource}" # Unique key for Access actions
            if not self.queue.exists(issue_number, queue_key):
                retry_count = self.queue.retry_count(issue_number, queue_key)
                self.queue.submit(
                    agent="access",
                    issue_number=issue_number,
                    payload={
                        "target_user_group": target_user_group,
                        "resource": resource,
                        "proposed_action": proposed_action,
                        "reason": reason,
                        "confidence_score": confidence_score,
                        "retry_count": retry_count,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    },
                    summary=summary
                )
                logger.info(f"Issue #{issue_number}: Access action submitted to approval queue.")
            else:
                logger.info(f"Issue #{issue_number}: Access action already in approval queue. Skipping re-submission.")

            return {
                "agent": "access",
                "status": "awaiting_approval",
                "target_user_group": target_user_group,
                "resource": resource,
                "llm_summary": summary,
                "auto_approved": False
            }

        except json.JSONDecodeError as e:
            error_msg = f"LLM response was not valid JSON. Error: {e}\nRaw response: {llm_response_content[:500]}..."
            logger.error(f"❌ AccessAgent JSON parsing failed for issue #{issue_number}: {error_msg}", exc_info=True)
            self.gh.add_comment(issue_number, f"⚠️ Access Agent: Failed to parse LLM response. {error_msg}")
            return {"agent": "access", "status": "failed", "error": error_msg}
        except Exception as e:
            error_msg = f"AccessAgent encountered an unexpected error: {e}"
            logger.error(f"❌ {error_msg} for issue #{issue_number}", exc_info=True)
            self.gh.add_comment(issue_number, f"⚠️ Access Agent: Remediation process failed. Error: `{e}`")
            return {"agent": "access", "status": "failed", "error": str(e)}

    def execute_remediation(self, issue_number: int, target_user_group: str, resource: str, proposed_action: str) -> Dict[str, Any]:
        """
        Simulates the execution of a proposed Access remediation.
        In a real-world scenario, this would interact with IAM systems, LDAP,
        or other access control mechanisms.

        Args:
            issue_number (int): The GitHub issue number.
            target_user_group (str): The user or group whose access needs modification.
            resource (str): The resource for which access is being modified.
            proposed_action (str): The specific action to take (e.g., 'grant read access', 'revoke ssh key').

        Returns:
            Dict[str, Any]: Status of the execution.
        """
        logger.info(f"Attempting to execute Access remediation for Issue #{issue_number}: Action '{proposed_action}' for '{target_user_group}' on '{resource}'.")
        msg = ""
        try:
            # Placeholder for actual Access Management automation.
            # E.g., calling an internal API to modify IAM roles, update user groups.
            # if "grant" in proposed_action.lower():
            #     iam_client.grant_access(target_user_group, resource, proposed_action)
            # elif "revoke" in proposed_action.lower():
            #     ldap_client.revoke_access(target_user_group, resource)

            # Simulate success for demonstration
            msg = (
                f"✅ **Access Action Executed**:\n"
                f"Simulated execution of '{proposed_action}' for '{target_user_group}' on '{resource}'.\n"
                f"In a real system, this would interact with your access management system."
            )
            logger.info(f"Successfully simulated Access action for issue #{issue_number}.")
            
            return {"status": "resolved", "message": msg, "auto_approved": True}
        except Exception as e:
            msg = f"❌ **Access Remediation Failed**:\n" \
                  f"Could not execute '{proposed_action}' for '{target_user_group}' on '{resource}'.\n" \
                  f"Error: `{e}`\n" \
                  f"Manual intervention by IT Security/Support required."
            logger.error(f"Failed to execute Access action for issue #{issue_number}: {e}", exc_info=True)
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
    logger.info("Starting AccessAgent example...")

    class MockIssue:
        def __init__(self, number, title, body):
            self.number = number
            self.title = title
            self.body = body

    mock_log_data = {
        "full_log": "ACCESS_DENIED - User 'johndoe' tried to access S3 bucket 'confidential-data' but was denied.",
        "error_summary": "Access denied for S3 bucket.",
        "user": "johndoe",
        "resource_type": "S3 Bucket",
        "resource_name": "confidential-data"
    }

    mock_enrichment_data = {
        "faiss_docs": [
            type('MockDoc', (object,), {
                'page_content': 'Permission denied error when accessing S3 bucket. Ensure user has appropriate IAM policy attached.',
                'metadata': {'category': 'access', 'root_cause': 'missing_iam_policy', 'solution': 'Attach S3 read-only policy to user/role.'}
            })()
        ],
        "similar_issues_graph": [
            {"issueNumber": 301, "title": "S3 access denied for analytics team", "category": "access"}
        ],
        "related_neo4j_solutions": [
            {"solutionName": "Review IAM policies", "solutionDescription": "Examine the specific IAM policies applied to the user or role."}
        ]
    }

    access_agent = AccessAgent()

    print("\n--- Testing handle method for Access issue ---")
    mock_issue_access = MockIssue(
        number=302,
        title="Access Request: User needs S3 bucket access",
        body="User 'janedoe' is requesting read access to 'reporting-data' S3 bucket for new project. Current access denied."
    )
    handle_result_access = access_agent.handle(mock_issue_access, mock_log_data, mock_enrichment_data)
    print(json.dumps(handle_result_access, indent=2))

    print("\n--- Simulating execute_remediation for Access issue ---")
    if handle_result_access.get("status") == "awaiting_approval":
        execute_result_access = access_agent.execute_remediation(
            issue_number=handle_result_access["issue_number"],
            target_user_group=handle_result_access.get("target_user_group", "janedoe"),
            resource=handle_result_access.get("resource", "reporting-data"),
            proposed_action=handle_result_access.get("llm_summary", "").split('\n')[2].replace('**Proposed Action**: ', '')
        )
        print(json.dumps(execute_result_access, indent=2))

    logger.info("AccessAgent example finished.")
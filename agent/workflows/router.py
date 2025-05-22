# agent/workflows/agents/router.py
import logging
import json
from typing import List, Dict, Any, Tuple

from agent.workflows.config import settings
from agent.rag.vector_store import Document # Import Document for type hinting
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

# Configure logging for this module
logger = logging.getLogger(__name__)

# Import the correct LLM library based on settings.LLM_MODEL
if "gemini" in settings.LLM_MODEL:
    import google.generativeai as genai
    # Ensure GOOGLE_API_KEY is set in your .env
    genai.configure(api_key=settings.GOOGLE_API_KEY.get_secret_value())
    LLM_CLIENT = genai.GenerativeModel(settings.LLM_MODEL)
    logger.info(f"Router: Initialized Google Generative AI client with model: {settings.LLM_MODEL}")
else:
    from langchain_openai import ChatOpenAI
    LLM_CLIENT = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=0, # Router should be deterministic
        openai_api_key=settings.OPENAI_API_KEY.get_secret_value()
    )
    logger.info(f"Router: Initialized Langchain ChatOpenAI client with model: {settings.LLM_MODEL}")


# Define explicit categories for routing
ROUTING_CATEGORIES = ["airflow", "devops", "access", "unclassified"]

# Prompt for the LLM-powered router
ROUTING_PROMPT = """
You are an intelligent issue routing system. Your task is to analyze a GitHub issue and its associated context (logs, historical similar issues from a vector store and graph database) and determine which specialized agent should handle it.

Your decision should be based on the core problem described in the issue, the nature of the logs, and the context provided by the RAG system.

Here are the available categories for routing:
- **airflow**: Issues related to Airflow DAGs, tasks, scheduler, or specific Airflow components. Often involves DAG failures, task retries, data pipeline issues.
- **devops**: Issues related to infrastructure (Kubernetes, VMs, cloud services), general system alerts, deployments, or network problems.
- **access**: Issues related to user permissions, access control, authentication, or authorization for any system.
- **unclassified**: If you cannot confidently categorize the issue into one of the above.

Provide your output as a JSON object with two keys:
- `category`: The chosen category (one of 'airflow', 'devops', 'access', 'unclassified').
- `reason`: A brief explanation (1-2 sentences) for why you chose that category.

---
GitHub Issue Title: {issue_title}
GitHub Issue Body: {issue_body}

---
Relevant Log Excerpt (if available):
{full_log}

---
Historical Documents (from Vector Store):
{faiss_docs_context}

---
Similar Issues (from Graph Database):
{graph_issues_context}

---
Based on the above information, classify this issue:
"""

def determine_routing(
    issue_text: str,
    full_log: str,
    faiss_docs: List[Document],
    similar_issues_graph: List[Dict[str, Any]]
) -> Tuple[str, str]:
    """
    Determines the appropriate agent to route a GitHub issue to using an LLM,
    leveraging RAG context.

    Args:
        issue_text (str): Combined title and body of the GitHub issue.
        full_log (str): The full log content fetched for the issue.
        faiss_docs (List[Document]): Relevant documents retrieved from the vector store (FAISS).
        similar_issues_graph (List[Dict[str, Any]]): Similar issues found in the Neo4j graph.

    Returns:
        Tuple[str, str]: A tuple containing the determined category (e.g., "airflow", "devops")
                         and the LLM's reasoning for the classification.
    """
    logger.info(f"Router: Determining routing for issue: '{issue_text[:50]}...'")

    # Prepare context for the prompt
    issue_title_parsed = issue_text.split('\n', 1)[0]
    issue_body_parsed = issue_text.split('\n', 1)[1] if '\n' in issue_text else ""

    faiss_context_str = "\n".join([f"Content: {doc.page_content}\nMetadata: {doc.metadata}" for doc in faiss_docs]) if faiss_docs else "No relevant documents found."
    graph_context_str = json.dumps(similar_issues_graph, indent=2) if similar_issues_graph else "No similar issues found in graph."

    prompt_content = ROUTING_PROMPT.format(
        issue_title=issue_title_parsed,
        issue_body=issue_body_parsed,
        full_log=full_log if full_log else "No full log available.",
        faiss_docs_context=faiss_context_str,
        graph_issues_context=graph_context_str
    )

    parser = JsonOutputParser()

    try:
        if "gemini" in settings.LLM_MODEL:
            # For Google Generative AI (Gemini)
            response = LLM_CLIENT.generate_content(prompt_content)
            llm_response_content = response.candidates[0].content.parts[0].text
        else:
            # For Langchain (OpenAI, etc.)
            messages = [
                SystemMessage(content="You are an intelligent issue routing system. Provide JSON output."),
                HumanMessage(content=prompt_content),
            ]
            llm_response_content = LLM_CLIENT.invoke(messages).content

        parsed_response = parser.parse(llm_response_content)

        category = parsed_response.get("category", "unclassified").lower()
        reason = parsed_response.get("reason", "LLM provided no specific reason.")

        if category not in ROUTING_CATEGORIES:
            logger.warning(f"Router: LLM returned an invalid category '{category}'. Defaulting to 'unclassified'.")
            category = "unclassified"
            reason = f"Original LLM category '{parsed_response.get('category')}' was invalid. Reason: {reason}"

        logger.info(f"Router: Classified as '{category}' because: {reason}")
        return category, reason

    except json.JSONDecodeError as e:
        logger.error(f"Router: LLM response was not valid JSON. Defaulting to 'unclassified'. Error: {e}\nRaw response: {llm_response_content[:500]}...", exc_info=True)
        return "unclassified", f"LLM response JSON parsing failed: {e}. Raw response: {llm_response_content[:100]}..."
    except Exception as e:
        logger.error(f"Router: An unexpected error occurred during routing: {e}. Defaulting to 'unclassified'.", exc_info=True)
        return "unclassified", f"An unexpected error occurred during LLM routing: {e}"

# Example Usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Starting Router example...")

    # Mock data for testing
    mock_issue_text_airflow = "Airflow DAG `my_etl` failed at task `transform_data` due to a ValueError."
    mock_log_airflow = "ERROR - ValueError: Invalid data format in transform_data task."
    mock_faiss_airflow = [
        Document(page_content="Airflow task failure, ValueError, data transformation. Solution: Validate input schema.", metadata={"category": "airflow", "root_cause": "data_schema_mismatch"}),
        Document(page_content="DAG `another_etl` timeout issue. Solution: Increase task timeout.", metadata={"category": "airflow", "root_cause": "timeout"})
    ]
    mock_graph_airflow = [
        {"issueNumber": 10, "title": "DAG `data_pipeline` failed on data validation", "category": "airflow"},
        {"issueNumber": 12, "title": "Airflow task `clean_data` stuck", "category": "airflow"}
    ]

    mock_issue_text_devops = "Kubernetes pod 'web-app-v2' stuck in CrashLoopBackOff. Logs indicate permission issues."
    mock_log_devops = "ERROR - Failed to mount volume: access denied for /var/log/app."
    mock_faiss_devops = [
        Document(page_content="Kubernetes pod crash, volume mount error, permission denied. Solution: Check K8s RBAC.", metadata={"category": "devops", "root_cause": "k8s_permissions"}),
        Document(page_content="Node out of disk space. Solution: Clean old logs.", metadata={"category": "devops", "root_cause": "disk_full"})
    ]
    mock_graph_devops = [
        {"issueNumber": 25, "title": "K8s deployment failing on startup", "category": "devops"},
        {"issueNumber": 28, "title": "Infrastructure resource exhaustion", "category": "devops"}
    ]

    mock_issue_text_access = "User 'alice' cannot SSH into production server. Permission denied."
    mock_log_access = "AUTH_FAIL - User alice attempted to log in, but public key not recognized."
    mock_faiss_access = [
        Document(page_content="SSH access denied, public key issue. Solution: Add user's public key to authorized_keys.", metadata={"category": "access", "root_cause": "ssh_key_missing"}),
        Document(page_content="API key invalid for service account. Solution: Regenerate API key.", metadata={"category": "access", "root_cause": "api_key_expired"})
    ]
    mock_graph_access = [
        {"issueNumber": 40, "title": "Production access request for new hire", "category": "access"},
        {"issueNumber": 42, "title": "VPN connection issues", "category": "access"}
    ]
    
    mock_issue_text_unclassified = "General system slowness observed. No specific errors or logs."
    mock_log_unclassified = "No clear error patterns."
    mock_faiss_unclassified = []
    mock_graph_unclassified = []


    print("\n--- Testing Airflow Routing ---")
    category_airflow, reason_airflow = determine_routing(
        mock_issue_text_airflow, mock_log_airflow, mock_faiss_airflow, mock_graph_airflow
    )
    print(f"Routed to: {category_airflow}, Reason: {reason_airflow}")

    print("\n--- Testing DevOps Routing ---")
    category_devops, reason_devops = determine_routing(
        mock_issue_text_devops, mock_log_devops, mock_faiss_devops, mock_graph_devops
    )
    print(f"Routed to: {category_devops}, Reason: {reason_devops}")

    print("\n--- Testing Access Routing ---")
    category_access, reason_access = determine_routing(
        mock_issue_text_access, mock_log_access, mock_faiss_access, mock_graph_access
    )
    print(f"Routed to: {category_access}, Reason: {reason_access}")

    print("\n--- Testing Unclassified Routing ---")
    category_unclassified, reason_unclassified = determine_routing(
        mock_issue_text_unclassified, mock_log_unclassified, mock_faiss_unclassified, mock_graph_unclassified
    )
    print(f"Routed to: {category_unclassified}, Reason: {reason_unclassified}")


    logger.info("Router example finished.")
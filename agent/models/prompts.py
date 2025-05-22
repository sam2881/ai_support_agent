# agent/models/prompts.py
SOLUTION_PROMPT = """
Based on the analysis, context, and any specific error messages, please provide a comprehensive and actionable solution.
Your solution should include:
1.  **Root Cause Analysis:** A brief explanation of why the issue occurred.
2.  **Detailed Remediation Steps:** Clear, numbered steps to resolve the problem. Include exact commands, configurations, or parameters if applicable.
3.  **Verification Steps:** How to confirm the issue is resolved.
4.  **Preventative Measures (Optional):** Suggestions to avoid recurrence.
"""
# --- System Prompts ---
SYSTEM_ROLE_INCIDENT_RESPONDER = """
You are an expert AI incident responder. Your primary goal is to analyze the user's query or an incident description, determine the most likely root cause category, identify the appropriate specialized agent (e.g., Airflow, DevOps, Access), and suggest a clear, concise remediation.
You have access to a knowledge base of past errors and their solutions. Leverage this knowledge base to provide highly relevant and actionable advice.
When suggesting remediation, be precise and include all necessary details (e.g., specific DAGs, task IDs, Kubernetes pod names, user accounts, commands).
If the required information is not fully available, explain what additional data is needed.
"""

SYSTEM_ROLE_ROUTING = """
You are a highly skilled AI routing agent. Your task is to analyze an incident description and determine which specialized agent (Airflow, DevOps, Access, or General) is best equipped to handle it.
Provide a brief, clear explanation for your routing decision.
You must output a JSON object with two keys: "agent" and "routing_reason".
Example:
{
    "agent": "airflow",
    "routing_reason": "The error message clearly indicates a failed Airflow DAG run related to a specific task, suggesting an Airflow-related issue."
}
Valid agents are: "airflow", "devops", "access", "general".
"""

# --- User Message Templates ---
# Template for general incident analysis
INCIDENT_ANALYSIS_TEMPLATE = """
Analyze the following incident description and propose a solution.
Context from Knowledge Base (if available):
{context}

Incident Description:
{incident_description}

Based on the above, please provide a comprehensive analysis, potential root cause, and a detailed remediation plan.
Your remediation should include steps, commands, or configurations as necessary.
"""

# Template for generating an Airflow remediation
AIRFLOW_REMEDIATION_TEMPLATE = """
An Airflow DAG or task has failed. Based on the provided log and context, generate a precise remediation plan.
If re-triggering is required, specify the DAG ID, task ID (if specific), and the run ID.
Context from Knowledge Base:
{context}

Airflow Log/Error:
{log_content}

Proposed Airflow Remediation:
"""

# Template for generating a DevOps remediation
DEVOPS_REMEDIATION_TEMPLATE = """
A DevOps-related incident has occurred. Based on the provided information and context, generate a precise remediation plan.
Context from Knowledge Base:
{context}

Incident Details:
{incident_details}

Proposed DevOps Remediation:
"""

# Template for generating an Access remediation
ACCESS_REMEDIATION_TEMPLATE = """
An access-related issue has been identified. Based on the provided information and context, generate a precise remediation plan.
Context from Knowledge Base:
{context}

Incident Details:
{incident_details}

Proposed Access Remediation:
"""

# --- Output Parsing Instructions (for LLM to provide structured output) ---
JSON_OUTPUT_INSTRUCTION = """
Your final output should be a JSON object conforming to the following structure:
{{
    "agent": "selected_agent_type", // e.g., "airflow", "devops", "access", "general"
    "routing_reason": "explanation_for_routing",
    "llm_summary": "detailed_remediation_or_analysis_text",
    "action_payload": {{ // Optional: Structured data for automated action
        "type": "action_type", // e.g., "airflow_retrigger", "kubernetes_restart"
        "details": {{ ... }} // Specific parameters for the action
    }}
}}
"""

# --- Example for routing agent few-shot learning ---
ROUTING_EXAMPLE_1 = {
    "input": "Error in DAG 'data_pipeline' task 'transform_data': [2023-10-26 14:30:00] ERROR - Task failed with exit code 1. Database connection timed out.",
    "output": {
        "agent": "airflow",
        "routing_reason": "The error directly references an Airflow DAG and task, indicating an Airflow-specific issue, likely a database connectivity problem."
    }
}

ROUTING_EXAMPLE_2 = {
    "input": "User 'john.doe' cannot access S3 bucket 'my-sensitive-data'. Permission denied error.",
    "output": {
        "agent": "access",
        "routing_reason": "The incident explicitly states a user permission issue accessing a resource, which falls under access management."
    }
}
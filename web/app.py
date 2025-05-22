from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from agent.clients.github_client import GitHubClient
from agent.workflows.main_agent import MainAgent
from agent.workflows.agents.approval_processor import ApprovalProcessor
from agent.workflows.config import settings # Import settings
from agent.utils.queue_utils import QUEUE_FILE, PROCESSED_LOG_FILE # Import file paths from queue_utils
import json
import os
import logging
from typing import Dict, Any

# Configure logging for the FastAPI application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Support Agent Console")
templates = Jinja2Templates(directory="templates")

# Initialize clients and agents once
try:
    gh_client = GitHubClient()
    main_agent = MainAgent()
    approval_processor = ApprovalProcessor() # Initialize ApprovalProcessor here
    logger.info("Successfully initialized GitHubClient and MainAgent.")
except Exception as e:
    logger.critical(f"Failed to initialize core components: {e}", exc_info=True)
    # Depending on severity, you might want to exit or disable functionalities.
    # For now, just log and allow app to start, but subsequent calls might fail.


@app.get("/", response_class=HTMLResponse)
async def read_ui(request: Request):
    """Renders the main UI page with a list of open GitHub issues."""
    try:
        issues = gh_client.repo.get_issues(state="open")
        logger.info(f"Fetched {len(list(issues))} open GitHub issues for UI.")
        return templates.TemplateResponse("index.html", {"request": request, "issues": issues})
    except Exception as e:
        logger.error(f"Error fetching GitHub issues for UI: {e}", exc_info=True)
        # Return a user-friendly error page or message
        return templates.TemplateResponse("error.html", {"request": request, "message": "Failed to load issues. Please check agent configuration and GitHub API access."}, status_code=500)


@app.get("/issues", response_class=JSONResponse)
async def get_github_issues():
    """Returns a JSON list of open GitHub issues."""
    try:
        issues = gh_client.repo.get_issues(state="open")
        return JSONResponse([{"number": i.number, "title": i.title} for i in issues])
    except Exception as e:
        logger.error(f"Error fetching GitHub issues via API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve GitHub issues.")


@app.get("/issues-options", response_class=HTMLResponse)
async def get_issues_dropdown():
    """Returns HTML options for a dropdown list of open GitHub issues."""
    try:
        issues = gh_client.repo.get_issues(state="open")
        html_options = "\n".join(
            f"<option value='{issue.number}'>#{issue.number} - {issue.title}</option>"
            for issue in issues
        )
        return HTMLResponse(html_options)
    except Exception as e:
        logger.error(f"Error generating issues dropdown: {e}", exc_info=True)
        return HTMLResponse("<option value=''>Error loading issues</option>", status_code=500)


@app.post("/process-ui", response_class=HTMLResponse)
async def process_issue_ui(request: Request, issue_number: int = Form(...)):
    """
    Processes a GitHub issue using the MainAgent and returns a formatted HTML response
    with the processing results.
    """
    logger.info(f"Received request to process GitHub issue #{issue_number} from UI.")
    try:
        # Process the issue with the MainAgent
        result = main_agent.process_issue(issue_number)
        logger.info(f"MainAgent finished processing issue #{issue_number}.")

        # Decide when to close the issue.
        # If approval is needed before remediation, don't close it here.
        # If the agent's proposal is the final step, then closing here is fine.
        # For now, keeping your original logic to close immediately after processing.
        try:
            issue = gh_client.repo.get_issue(number=issue_number)
            if issue.state == "open": # Only close if it's still open
                issue.edit(state="closed")
                logger.info(f"Closed GitHub issue #{issue_number} after processing.")
            else:
                logger.info(f"GitHub issue #{issue_number} was already {issue.state}.")
        except Exception as close_err:
            logger.warning(f"Failed to close GitHub issue #{issue_number}: {close_err}", exc_info=True)

        # Construct the HTML response with detailed steps
        return HTMLResponse(
            f"""
            <div class="space-y-4">
                <div class="bg-blue-50 border border-blue-200 p-4 rounded-lg shadow-sm">
                    <h3 class="font-semibold text-lg text-blue-800 mb-2">📌 Step 2: Routing Explanation</h3>
                    <p class="text-sm text-blue-700 whitespace-pre-wrap">
                        {result.get("routing_reason", "No routing details available from the agent.")}
                    </p>
                </div>
                <div class="bg-purple-50 border border-purple-200 p-4 rounded-lg shadow-sm">
                    <h3 class="font-semibold text-lg text-purple-800 mb-2">🔀 Step 3: Routed Agent</h3>
                    <p class="text-sm text-purple-700">
                        Agent selected: <code class="font-mono bg-purple-100 px-1 py-0.5 rounded text-purple-900">{result.get("agent", "Unrouted")}</code>
                    </p>
                </div>
                <div class="bg-gray-50 border border-gray-200 p-4 rounded-lg shadow-sm">
                    <h3 class="font-semibold text-lg text-gray-800 mb-2">💡 Step 4: LLM Suggestion</h3>
                    <pre class="text-sm text-gray-700 bg-gray-100 p-3 rounded-md overflow-x-auto whitespace-pre-wrap"><code>
{result.get("llm_summary", "The AI did not provide a specific suggestion or summary for this issue.")}
                    </code></pre>
                </div>
                <div class="bg-yellow-50 border border-yellow-200 p-4 rounded-lg shadow-sm">
                    <h3 class="font-semibold text-lg text-yellow-800 mb-2">✔️ Step 5: Approval Required</h3>
                    <p class="text-sm text-yellow-700">
                        Proposed actions have been added to the approval queue. Awaiting human review and execution.
                    </p>
                </div>
            </div>
            """
        )
    except Exception as e:
        logger.error(f"Error processing GitHub issue #{issue_number}: {e}", exc_info=True)
        return HTMLResponse(
            f"""
            <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
                <strong class="font-bold">Error!</strong>
                <span class="block sm:inline">Failed to process issue #{issue_number}. An internal error occurred.</span>
                <p class="text-sm mt-1">Details: {e}</p>
            </div>
            """,
            status_code=500
        )


@app.post("/approve-ui", response_class=HTMLResponse)
async def approve_ui(request: Request):
    """
    Processes all pending approvals in the queue and triggers remediation actions.
    """
    logger.info("Received request to process approval queue from UI.")
    try:
        approval_processor.process_approvals()
        logger.info("Approval queue processing completed.")
        return HTMLResponse(
            "<div class='bg-green-50 border border-green-300 text-green-800 p-4 rounded'>✅ Approval queue processed and proposed actions executed.</div>"
        )
    except Exception as e:
        logger.error(f"Error processing approval queue: {e}", exc_info=True)
        return HTMLResponse(
            f"<div class='bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative'>❌ Failed to process approval queue: {e}</div>",
            status_code=500
        )


@app.get("/pending-approvals", response_class=JSONResponse)
async def get_pending_approvals():
    """Returns a JSON list of items currently in the pending approval queue."""
    approvals = []
    try:
        # Use QUEUE_FILE from queue_utils
        if not QUEUE_FILE.exists():
            return JSONResponse(content=[])
        
        with QUEUE_FILE.open("r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    approvals.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON in {QUEUE_FILE} at line {line_num}: {e}. Line: '{line.strip()}'")
                except Exception as e:
                    logger.error(f"Unexpected error reading line {line_num} from {QUEUE_FILE}: {e}", exc_info=True)
        logger.info(f"Fetched {len(approvals)} pending approvals.")
    except IOError as e:
        logger.error(f"Error reading pending approvals file {QUEUE_FILE}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read pending approvals file.")
    return JSONResponse(content=approvals)


@app.get("/executed-approvals", response_class=JSONResponse)
async def get_executed_approvals():
    """Returns a JSON list of items that have been executed from the approval queue."""
    approvals = []
    try:
        # Use PROCESSED_LOG_FILE from queue_utils
        if not PROCESSED_LOG_FILE.exists():
            return JSONResponse(content=[])
        
        with PROCESSED_LOG_FILE.open("r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    approvals.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON in {PROCESSED_LOG_FILE} at line {line_num}: {e}. Line: '{line.strip()}'")
                except Exception as e:
                    logger.error(f"Unexpected error reading line {line_num} from {PROCESSED_LOG_FILE}: {e}", exc_info=True)
        logger.info(f"Fetched {len(approvals)} executed approvals.")
    except IOError as e:
        logger.error(f"Error reading executed approvals file {PROCESSED_LOG_FILE}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read executed approvals file.")
    return JSONResponse(content=approvals)


@app.get("/executed-approvals-ui", response_class=HTMLResponse)
async def view_executed_approvals():
    """Renders an HTML snippet of executed approvals."""
    rows = []
    try:
        # Use PROCESSED_LOG_FILE from queue_utils
        if not PROCESSED_LOG_FILE.exists():
            return HTMLResponse("<div class='text-gray-500'>⚠️ No approvals executed yet.</div>")

        with PROCESSED_LOG_FILE.open("r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item: Dict[str, Any] = json.loads(line.strip())
                    payload = item.get('payload', {})
                    
                    # Generic display for payload, to handle different agent types
                    payload_details = []
                    if item.get("agent") == "airflow":
                        payload_details.append(f"DAG: <code class='font-mono'>{payload.get('dag_id', 'N/A')}</code>")
                        payload_details.append(f"Task: <code class='font-mono'>{payload.get('task_id', 'N/A')}</code>")
                        payload_details.append(f"Run ID: <code class='font-mono'>{payload.get('dag_run_id', 'N/A')}</code>")
                        payload_details.append(f"Action: <code class='font-mono'>{payload.get('action', 'N/A')}</code>")
                    elif item.get("agent") == "devops":
                        payload_details.append(f"Target: <code class='font-mono'>{payload.get('target_component', 'N/A')}</code>")
                        payload_details.append(f"Action: <code class='font-mono'>{payload.get('proposed_action', 'N/A')}</code>")
                    elif item.get("agent") == "access":
                        payload_details.append(f"User/Group: <code class='font-mono'>{payload.get('user_or_group', 'N/A')}</code>")
                        payload_details.append(f"Resource: <code class='font-mono'>{payload.get('resource_name', 'N/A')}</code>")
                        payload_details.append(f"Action: <code class='font-mono'>{payload.get('action_type', 'N/A')}</code>")
                    else:
                        payload_details.append(f"Payload: <code class='font-mono'>{json.dumps(payload)}</code>") # Fallback for unknown agents

                    rows.append(f"""
                        <div class='bg-gray-50 border border-gray-200 p-3 rounded-lg mb-2 shadow-sm'>
                            <p class='text-sm text-gray-800 mb-1'>
                                ✅ <strong>Issue #{item.get('issue_number', 'N/A')}</strong> (Agent: <span class='font-semibold text-indigo-700'>{item.get('agent', 'N/A')}</span>)<br>
                                {' | '.join(payload_details)}<br>
                                Proposed Summary: <span class='italic'>"{item.get('summary', 'No summary provided')}"</span><br>
                                Processed Timestamp: <span class='italic'>{item.get('processed_timestamp', 'n/a')}</span>
                            </p>
                        </div>
                    """)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON in {PROCESSED_LOG_FILE} at line {line_num} for UI: {e}. Line: '{line.strip()}'")
                except Exception as e:
                    logger.error(f"Unexpected error rendering executed approval line {line_num}: {e}", exc_info=True)

        if not rows:
            return HTMLResponse("<div class='text-gray-500'>⚠️ No approvals executed yet, or all records are malformed.</div>")

        return HTMLResponse("".join(rows))

    except IOError as e:
        logger.error(f"Error reading executed approvals file {PROCESSED_LOG_FILE} for UI: {e}", exc_info=True)
        return HTMLResponse(f"<div class='bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative'>❌ Failed to load executed approvals for display: {e}</div>", status_code=500)
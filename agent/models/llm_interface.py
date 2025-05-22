import os
import json
import logging
from typing import List, Dict, Any, Optional

# Import the prompts (assuming they are correctly defined in agent/models/prompts.py)
from agent.models.prompts import (
    SYSTEM_ROLE_ROUTING, SYSTEM_ROLE_INCIDENT_RESPONDER,
    INCIDENT_ANALYSIS_TEMPLATE, AIRFLOW_REMEDIATION_TEMPLATE,
    DEVOPS_REMEDIATION_TEMPLATE, ACCESS_REMEDIATION_TEMPLATE,
    JSON_OUTPUT_INSTRUCTION,
    ROUTING_EXAMPLE_1, ROUTING_EXAMPLE_2 # For few-shot
)
from agent.workflows.config import settings# Assuming settings handles API keys

logger = logging.getLogger(__name__)

# Try to import OpenAI client
try:
    from openai import OpenAI, APIError
    logger.info("OpenAI client imported successfully.")
except ImportError:
    OpenAI = None
    APIError = None
    logger.warning("OpenAI client not found. Install 'openai' if you want to use OpenAI models.")

# Try to import Google Generative AI client
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    logger.info("Google Generative AI client imported successfully.")
except ImportError:
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None
    logger.warning("Google Generative AI client not found. Install 'google-generativeai' if you want to use Gemini models.")


class LLMInterface:
    """
    A unified interface for interacting with various Large Language Models (LLMs).
    Supports OpenAI and Google Gemini.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name if model_name else settings.DEFAULT_LLM_MODEL
        self.client: Any = None
        self._initialize_client()

    def _initialize_client(self):
        """Initializes the appropriate LLM client based on model_name."""
        logger.info(f"Initializing LLM client for model: {self.model_name}")

        if self.model_name.startswith("gpt"):
            if not OpenAI:
                raise ImportError("OpenAI client not available. Please install 'openai'.")
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in environment or config.")
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY.get_secret_value())
            logger.info("OpenAI client initialized.")
        elif self.model_name.startswith("gemini"):
            if not genai:
                raise ImportError("Google Generative AI client not available. Please install 'google-generativeai'.")
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set in environment or config.")
            genai.configure(api_key=settings.GEMINI_API_KEY.get_secret_value())
            self.client = genai
            logger.info("Google Generative AI client initialized.")
        else:
            raise ValueError(f"Unsupported LLM model name: {self.model_name}. "
                             "Please choose 'gpt' or 'gemini' based models.")

    def _build_messages_for_openai(self, system_prompt: str, user_prompt: str, examples: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        Builds messages in OpenAI chat completion format.
        Few-shot examples are inserted between system and user prompts.
        """
        messages = [{"role": "system", "content": system_prompt}]

        if examples:
            for example in examples:
                messages.append({"role": "user", "content": example["input"]})
                messages.append({"role": "assistant", "content": example["output"]})

        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _generate_content_openai(self, messages: List[Dict[str, str]], temperature: float) -> str:
        """Sends content to OpenAI chat completion API."""
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return chat_completion.choices[0].message.content
        except APIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI call: {e}", exc_info=True)
            raise

    def _generate_content_gemini(self, system_prompt: str, user_prompt: str, examples: Optional[List[Dict[str, Any]]] = None, temperature: float = 0.5) -> str:
        """Sends content to Google Gemini API."""
        try:
            model = genai.GenerativeModel(self.model_name)
            contents = []

            # Step 1: Add the system prompt as the beginning of the first user turn.
            # This is effective for guiding Gemini's behavior.
            initial_user_message_parts = [{"text": system_prompt + "\n\n"}]

            # Step 2: Add few-shot examples (if any) as alternating user/model turns.
            if examples:
                # Add the initial user message part to the contents list
                # This ensures the system prompt is at the very beginning of the conversation.
                contents.append({"role": "user", "parts": initial_user_message_parts})

                for example in examples:
                    contents.append({
                        "role": "user",
                        "parts": [{"text": example["input"]}]
                    })
                    contents.append({
                        "role": "model",
                        "parts": [{"text": example["output"]}]
                    })
                
                # Step 3: Add the actual current user prompt as the final user turn.
                contents.append({
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                })
            else:
                # If no examples, just combine system_prompt and user_prompt into one user turn.
                initial_user_message_parts.append({"text": user_prompt})
                contents.append({"role": "user", "parts": initial_user_message_parts})


            generation_config = {
                "temperature": temperature,
                "response_mime_type": "application/json" # For forcing JSON output
            }

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            response = model.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            # Raise an error if the response was blocked by safety settings
            if not response.candidates:
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    raise Exception(f"Gemini prompt blocked due to safety settings: {response.prompt_feedback.block_reason}")
                raise Exception("Gemini generated no candidates (response likely blocked or empty).")

            return response.text

        except Exception as e:
            logger.error(f"An error occurred during Gemini call: {e}", exc_info=True)
            raise

    def get_routing_suggestion(self, incident_description: str) -> Dict[str, str]:
        """
        Gets a routing suggestion from the LLM based on the incident description.
        Returns a dictionary with 'agent' and 'routing_reason'.
        """
        user_prompt_content = f"Incident Description: {incident_description}"
        
        # Few-shot examples for routing (ensure outputs are JSON strings)
        examples = [
            {"input": ROUTING_EXAMPLE_1["input"], "output": json.dumps(ROUTING_EXAMPLE_1["output"])},
            {"input": ROUTING_EXAMPLE_2["input"], "output": json.dumps(ROUTING_EXAMPLE_2["output"])},
        ]

        # Combine with routing system prompt and JSON output instruction
        system_prompt = SYSTEM_ROLE_ROUTING
        final_user_prompt_for_llm = (
            f"{user_prompt_content}\n\n{JSON_OUTPUT_INSTRUCTION}"
        )
        
        logger.info(f"Getting routing suggestion for: {incident_description[:100]}...")

        try:
            llm_output = ""
            if self.model_name.startswith("gpt"):
                messages = self._build_messages_for_openai(system_prompt, final_user_prompt_for_llm, examples=examples)
                llm_output = self._generate_content_openai(messages, temperature=0.1)
            elif self.model_name.startswith("gemini"):
                # Call the corrected _generate_content_gemini
                llm_output = self._generate_content_gemini(
                    system_prompt=system_prompt,
                    user_prompt=final_user_prompt_for_llm,
                    examples=examples,
                    temperature=0.1
                )
            
            logger.debug(f"Raw LLM Routing Output: {llm_output}")
            parsed_output = self.parse_json_output(llm_output)
            
            if "agent" not in parsed_output or "routing_reason" not in parsed_output:
                logger.warning(f"LLM routing output missing required keys. Raw: {llm_output}")
                return {"agent": "general", "routing_reason": "LLM failed to provide specific routing."}
            
            return parsed_output
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM routing response: {e}. Raw: {llm_output}", exc_info=True)
            return {"agent": "general", "routing_reason": f"LLM output not valid JSON: {e}"}
        except Exception as e:
            logger.error(f"Error getting routing suggestion from LLM: {e}", exc_info=True)
            return {"agent": "general", "routing_reason": f"LLM routing error: {e}"}

    def get_remediation_suggestion(self, agent_type: str, incident_details: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Gets a detailed remediation suggestion from the LLM based on agent type and context.
        Returns a dictionary with 'llm_summary' and optionally 'action_payload'.
        """
        system_prompt = SYSTEM_ROLE_INCIDENT_RESPONDER
        user_prompt_template = ""

        if agent_type == "airflow":
            user_prompt_template = AIRFLOW_REMEDIATION_TEMPLATE
        elif agent_type == "devops":
            user_prompt_template = DEVOPS_REMEDIATION_TEMPLATE
        elif agent_type == "access":
            user_prompt_template = ACCESS_REMEDIATION_TEMPLATE
        else: # general or fallback
            user_prompt_template = INCIDENT_ANALYSIS_TEMPLATE

        # Format the user prompt with incident details and context
        formatted_user_prompt = user_prompt_template.format(
            context=context if context else "No relevant context found.",
            log_content=incident_details, # Using log_content or incident_details interchangeably
            incident_details=incident_details # Also for devops/access templates
        )
        
        # Combine with JSON output instruction
        final_user_prompt = f"{formatted_user_prompt}\n\n{JSON_OUTPUT_INSTRUCTION}"

        logger.info(f"Getting remediation suggestion for agent '{agent_type}' and incident: {incident_details[:100]}...")

        try:
            llm_output = ""
            if self.model_name.startswith("gpt"):
                messages = self._build_messages_for_openai(system_prompt, final_user_prompt, examples=None) # No examples for remediation
                llm_output = self._generate_content_openai(messages, temperature=0.5)
            elif self.model_name.startswith("gemini"):
                # Call the corrected _generate_content_gemini
                llm_output = self._generate_content_gemini(
                    system_prompt=system_prompt,
                    user_prompt=final_user_prompt,
                    examples=None, # No examples for remediation
                    temperature=0.5
                )

            logger.debug(f"Raw LLM Remediation Output: {llm_output}")
            parsed_output = self.parse_json_output(llm_output)

            if "llm_summary" not in parsed_output:
                logger.warning(f"LLM remediation output missing 'llm_summary'. Raw: {llm_output}")
                parsed_output["llm_summary"] = "LLM failed to provide a specific summary."
            
            return parsed_output
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM remediation response: {e}. Raw: {llm_output}", exc_info=True)
            return {"llm_summary": f"LLM output not valid JSON: {e}"}
        except Exception as e:
            logger.error(f"Error getting remediation suggestion from LLM: {e}", exc_info=True)
            return {"llm_summary": f"LLM remediation error: {e}"}

    def parse_json_output(self, llm_response: str) -> Dict[str, Any]:
        """
        Attempts to parse a JSON string from the LLM's response.
        Handles cases where LLM might wrap JSON in markdown code blocks.
        """
        # Remove common markdown code block wrappers
        if llm_response.strip().startswith("```json") and llm_response.strip().endswith("```"):
            json_str = llm_response.strip()[len("```json"):-len("```")].strip()
        elif llm_response.strip().startswith("```") and llm_response.strip().endswith("```"):
            json_str = llm_response.strip()[len("```"):-len("```")].strip()
        else:
            json_str = llm_response.strip()

        return json.loads(json_str)

# Example Usage (for testing purposes)
if __name__ == "__main__":
    from dotenv import load_dotenv
    from unittest.mock import patch # Needed for mocking settings during local test
    load_dotenv() # Load .env file for testing

    # Use actual settings if available, otherwise fallback for local testing
    class TestSettings:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-1.5-pro") # Default to Gemini for testing

    with patch('agent.models.llm_interface.settings', TestSettings()):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.info("Running LLMInterface test...")

        # --- Test OpenAI ---
        if TestSettings.OPENAI_API_KEY:
            logger.info("\n--- Testing OpenAI (gpt-4o-mini) ---")
            try:
                llm_openai = LLMInterface(model_name="gpt-4o-mini")
                
                routing_desc_openai = "User 'bob' cannot log in to Grafana, reports 'Incorrect username or password'."
                routing_result_openai = llm_openai.get_routing_suggestion(routing_desc_openai)
                logger.info(f"OpenAI Routing Result: {routing_result_openai}")

                remediation_details_openai = "Grafana login failed for 'bob'. User is in LDAP group 'monitoring-users'."
                remediation_context_openai = "Knowledge base suggests checking LDAP sync or resetting password."
                remediation_result_openai = llm_openai.get_remediation_suggestion(
                    "access", remediation_details_openai, remediation_context_openai
                )
                logger.info(f"OpenAI Remediation Result: {remediation_result_openai}")

            except (ValueError, ImportError, APIError) as e:
                logger.error(f"OpenAI test failed: {e}")
            except Exception as e:
                logger.error(f"An unexpected error during OpenAI test: {e}")
        else:
            logger.warning("Skipping OpenAI test: OPENAI_API_KEY not set.")

        # --- Test Gemini ---
        if TestSettings.GEMINI_API_KEY:
            logger.info("\n--- Testing Gemini (gemini-1.5-pro) ---")
            try:
                llm_gemini = LLMInterface(model_name="gemini-1.5-pro")

                routing_desc_gemini = "Kubernetes pod 'my-app-deployment-abc' is in CrashLoopBackOff. Logs show 'Error: OOMKilled'."
                routing_result_gemini = llm_gemini.get_routing_suggestion(routing_desc_gemini)
                logger.info(f"Gemini Routing Result: {routing_result_gemini}")

                remediation_details_gemini = "Pod 'my-app-deployment-abc' OOMKilled. Needs more memory."
                remediation_context_gemini = "Previous OOMKilled issues were resolved by increasing resource limits in YAML."
                remediation_result_gemini = llm_gemini.get_remediation_suggestion(
                    "devops", remediation_details_gemini, remediation_context_gemini
                )
                logger.info(f"Gemini Remediation Result: {remediation_result_gemini}")

            except (ValueError, ImportError) as e:
                logger.error(f"Gemini test failed: {e}")
            except Exception as e:
                logger.error(f"An unexpected error during Gemini test: {e}")
        else:
            logger.warning("Skipping Gemini test: GEMINI_API_KEY not set.")
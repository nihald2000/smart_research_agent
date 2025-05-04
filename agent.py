import google.generativeai as genai
import os
import re
from dotenv import load_dotenv
import requests # For fetching image from URL if needed
from PIL import Image # For handling image data if needed
import io

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

genai.configure(api_key=API_KEY)

# --- Configuration ---
# Use gemini-1.5-pro-latest if available and preferred for planning/synthesis
PLANNING_MODEL_NAME = "gemini-1.5-pro-latest"
# Fallback or for simpler tasks
# PLANNING_MODEL_NAME = "gemini-pro"
EXTRACTION_MODEL_NAME = "gemini-pro"
VISION_MODEL_NAME = "gemini-pro-vision"
SYNTHESIS_MODEL_NAME = "gemini-1.5-pro-latest" # Good for combining complex info

# Safety settings - adjust as needed, be cautious with disabling too much
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

generation_config = {
    "temperature": 0.3, # Lower temperature for more factual/predictable planning/extraction
    "top_p": 0.95,
    "top_k": 40,
    # "max_output_tokens": 2048, # Adjust as needed
}


class GaiaGeminiAgent:
    def __init__(self):
        self.planner_model = genai.GenerativeModel(
            PLANNING_MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        self.extraction_model = genai.GenerativeModel(
            EXTRACTION_MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings
            )
        self.vision_model = genai.GenerativeModel(
             VISION_MODEL_NAME,
            safety_settings=safety_settings # Vision might have different config needs
            )
        self.synthesis_model = genai.GenerativeModel(
            SYNTHESIS_MODEL_NAME,
            generation_config=generation_config, # Might want higher temp for creative synthesis
            safety_settings=safety_settings
        )
        self.history = [] # Store conversation turns or state
        self.intermediate_results = {} # Store results like {'ship_name': 'SS Ile de France', 'painting_fruits': [...]}

    def _call_gemini(self, model, prompt, is_vision=False, image_parts=None):
        """Helper function to call Gemini API and handle potential errors."""
        print(f"\n--- Calling Model: {model.model_name} ---")
        print(f"Prompt: {prompt[:200]}...") # Print truncated prompt
        if is_vision and not image_parts:
             return "Error: Vision model called without image parts."

        try:
            if is_vision:
                # Vision model takes prompt and image parts list
                contents = [prompt] + image_parts
                response = model.generate_content(contents, stream=False)
            else:
                 # Text models take just the prompt string
                response = model.generate_content(prompt, stream=False)

            # Handle potential blocks or empty responses
            if not response.parts:
                 # Check for safety feedback if blocked
                 if response.prompt_feedback and response.prompt_feedback.block_reason:
                     reason = response.prompt_feedback.block_reason
                     print(f"Warning: Prompt blocked due to {reason}")
                     return f"Error: Call blocked by safety filter ({reason})."
                 else:
                     print("Warning: Received empty response from model.")
                     return "Error: Received no content from the model."

            result_text = response.text
            print(f"Result: {result_text[:200]}...") # Print truncated result
            return result_text

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Consider more specific error handling based on google.api_core.exceptions
            return f"Error: API call failed - {e}"

    def _get_image_parts_from_url(self, image_url):
        """Fetches image from URL and prepares it for Gemini Vision API."""
        print(f"Fetching image from URL: {image_url}")
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status() # Raise error for bad responses (4xx or 5xx)

            # Check content type
            content_type = response.headers.get('content-type')
            if not content_type or not content_type.startswith('image/'):
                 print(f"Warning: URL content type ({content_type}) doesn't look like an image.")
                 # You might want stricter checking here based on expected types (jpeg, png, webp)
                 # For simplicity, we'll try to process it anyway.

            image_bytes = response.content
            image = Image.open(io.BytesIO(image_bytes))

            # Prepare image part for Gemini API
            # Gemini API expects a list of parts, usually [prompt_text, image_data]
            # image_data itself needs 'mime_type' and 'data'
            # Determine mime_type (important!)
            img_format = image.format
            if img_format == 'JPEG':
                mime_type = 'image/jpeg'
            elif img_format == 'PNG':
                mime_type = 'image/png'
            elif img_format == 'WEBP':
                 mime_type = 'image/webp'
            # Add more formats if needed (GIF, BMP, etc. - check Gemini Vision docs for support)
            else:
                 print(f"Warning: Unsupported image format {img_format}. Attempting to send as PNG.")
                 # Convert to PNG as a fallback? Risky. Better to error out.
                 # For now, let's just try sending the original bytes with a guess. This might fail.
                 # A better approach is to convert to a supported format if needed.
                 # mime_type = 'image/png' # Example fallback
                 # OR
                 return None, f"Error: Unsupported image format '{img_format}' at URL."


            image_part = {
                "mime_type": mime_type,
                "data": image_bytes
            }
            print(f"Successfully prepared image part (MIME type: {mime_type}).")
            return [image_part], None # Return list containing the part, and no error

        except requests.exceptions.RequestException as e:
            print(f"Error fetching image URL {image_url}: {e}")
            return None, f"Error: Could not fetch image from URL - {e}"
        except Exception as e:
             print(f"Error processing image from URL {image_url}: {e}")
             return None, f"Error: Could not process image data - {e}"

    def plan(self, query):
        """Asks the planning model to break down the query."""
        prompt = f"""Given the complex user query below, break it down into a sequence of simple, factual questions or image analysis tasks that need to be answered to arrive at the final solution. Identify the specific information needed at each step. Output ONLY a numbered list of these steps/questions.

User Query: "{query}"

Steps:
1. ...
2. ...
"""
        plan_text = self._call_gemini(self.planner_model, prompt)
        if plan_text.startswith("Error:"):
            return None, plan_text # Propagate error

        # Simple parsing of numbered list (can be made more robust)
        steps = [line.strip() for line in plan_text.strip().split('\n') if re.match(r"^\d+\.\s+", line.strip())]
        print(f"Generated Plan Steps: {steps}")
        return steps, None

    def execute_step(self, step_description):
        """Executes a single step, deciding whether it's text extraction or vision."""
        print(f"\n--- Executing Step: {step_description} ---")
        self.history.append({"role": "agent", "content": f"Executing step: {step_description}"})

        # Basic keyword check to see if it involves an image
        # This is brittle - a better approach would involve the planner explicitly stating the step type
        is_image_task = "image" in step_description.lower() or "painting" in step_description.lower() or "photo" in step_description.lower()

        if is_image_task:
            # --- Image Analysis Logic ---
            # 1. Try to extract image identifier (URL or name) from the step description
            #    This might require another LLM call or regex. For simplicity, assume planner provides it.
            #    Example: Planner step might be "Analyze image at URL [URL] for fruits."
            image_url_match = re.search(r"https?://\S+", step_description) # Basic URL regex
            image_url = image_url_match.group(0) if image_url_match else None
            image_name = None # Could try extracting name if no URL

            if not image_url:
                 # If no URL, try asking Gemini to find the image URL based on the description in the step
                 find_url_prompt = f"Find a public URL for the image described here: '{step_description}'. Only output the URL if found, otherwise say 'URL not found'."
                 potential_url = self._call_gemini(self.extraction_model, find_url_prompt)
                 if not potential_url.startswith("Error:") and potential_url.startswith("http"):
                     image_url = potential_url.strip()
                     print(f"Found potential image URL via search: {image_url}")
                 else:
                     result = f"Error: Could not find a URL for the image required in step: '{step_description}'"
                     self.history.append({"role": "system", "content": result})
                     return result

            # 2. Fetch image data from URL
            image_parts, error_msg = self._get_image_parts_from_url(image_url)
            if error_msg:
                 result = f"Error processing image for step '{step_description}': {error_msg}"
                 self.history.append({"role": "system", "content": result})
                 return result

            # 3. Formulate Vision Prompt
            vision_prompt = f"""Analyze the provided image based on this instruction: {step_description}. Focus ONLY on fulfilling this specific request. For example, if asked to list fruits clockwise, provide only that list.

Instruction: "{step_description}"

Analysis Result:
"""
            result = self._call_gemini(self.vision_model, vision_prompt, is_vision=True, image_parts=image_parts)

        else:
            # --- Text/Fact Extraction Logic ---
            extraction_prompt = f"""Based on your knowledge, answer the following question accurately and concisely. Focus only on providing the specific information requested.

Question: "{step_description}"

Answer:
"""
            result = self._call_gemini(self.extraction_model, extraction_prompt)

        # Store result (even if error) and update history
        step_key = re.sub(r'[^\w\s-]', '', step_description.lower()).replace(' ', '_')[:50] # Basic key generation
        self.intermediate_results[step_key] = result
        self.history.append({"role": "system", "content": f"Result for '{step_description}': {result}"})

        print(f"Step Result ('{step_key}'): {result}")
        return result # Return the direct result of the step


    def synthesize(self, original_query):
        """Combines intermediate results to answer the original query."""
        print("\n--- Synthesizing Final Answer ---")

        context = "Intermediate results gathered:\n"
        for key, value in self.intermediate_results.items():
            # Clean up key for better readability in prompt
            clean_key = key.replace('_', ' ').title()
            context += f"- {clean_key}: {value}\n"

        synthesis_prompt = f"""Given the original user query and the following intermediate results gathered by executing a plan, synthesize the final answer. Adhere strictly to any formatting requirements mentioned in the original query.

Original User Query: "{original_query}"

{context}

Final Answer:
"""
        final_answer = self._call_gemini(self.synthesis_model, synthesis_prompt)
        self.history.append({"role": "agent", "content": f"Final synthesized answer: {final_answer}"})
        return final_answer


    def run(self, query):
        """Runs the full planning, execution, and synthesis process."""
        self.history = [{"role": "user", "content": query}]
        self.intermediate_results = {} # Reset state

        # 1. Plan
        print("\n--- Generating Plan ---")
        steps, error = self.plan(query)
        if error:
            print(f"Planning failed: {error}")
            return f"Sorry, I couldn't generate a plan to answer your question. Error: {error}"
        if not steps:
             print("Planning failed: No steps generated.")
             return "Sorry, I couldn't generate a plan for this query."

        self.history.append({"role": "agent", "content": f"Generated Plan:\n{chr(10).join(steps)}"}) # Use chr(10) for newline

        # 2. Execute Plan Steps
        print("\n--- Executing Plan ---")
        all_steps_succeeded = True
        for i, step in enumerate(steps):
            print(f"\n--- Step {i+1}/{len(steps)} ---")
            step_result = self.execute_step(step)
            if step_result.startswith("Error:"):
                print(f"Step failed: {step_result}")
                # Option 1: Stop immediately
                # return f"Sorry, I encountered an error during execution. Step failed: {step_result}"
                # Option 2: Try to continue, synthesis might fail or be incomplete
                all_steps_succeeded = False
                # break # Or remove break to try all steps regardless of errors

        if not all_steps_succeeded:
             print("\nWarning: One or more execution steps failed. Attempting synthesis with available data.")

        # 3. Synthesize
        print("\n--- Synthesizing Result ---")
        final_answer = self.synthesize(query)

        return final_answer
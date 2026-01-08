

"""
Unified Caption Generation, Evaluation, and Optimization Script
This script performs both caption evaluation and optimization in one pass.

Usage:
    python caption_judge_optimize.py --input input.json --output output.json --threshold 8

Features:
    - Generates initial captions using VLM with few-shot prompting
    - Evaluates captions based on accuracy, completeness, detail, relevance, and clarity
    - Automatically optimizes captions scoring below the threshold
    - Supports both diagnosis and knowledge QA tasks
"""

import argparse
import json
import os
import re
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ========== Configuration ==========
os.environ["OPENAI_API_BASE"] = "YOUR_API_BASE_URL"
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# ========== Define Output Format for Evaluation ==========
evaluation_schemas = [
    ResponseSchema(
        name="rating",
        description="Score from 1 to 10 based on caption quality"
    ),
    ResponseSchema(
        name="reasoning",
        description="Brief explanation for the rating"
    ),
    ResponseSchema(
        name="suggestions",
        description="Specific suggestions for improvement"
    )
]
evaluation_parser = StructuredOutputParser.from_response_schemas(evaluation_schemas)
evaluation_format_instructions = evaluation_parser.get_format_instructions()

# ========== Evaluation System Prompt ==========
evaluation_system_template = """You are an expert evaluator for agricultural image captions. Please evaluate the quality of the image caption based on the following criteria:

## Evaluation Criteria
1. **Accuracy**: Correct identification of plant and disease/pest (or confirmation of health)
2. **Completeness**: Inclusion of all key elements (plant type, disease type, symptoms, severity)
3. **Detail**: Specific description of symptoms (location, shape, color, extent, quantity, etc.)
4. **Relevance**: Information is relevant for agricultural diagnosis and treatment
5. **Clarity**: Clear, concise, and professional language (80-120 words)

## Rating Scale
- 1-3: Poor (vague, inaccurate, or missing key information)
- 4-6: Fair (some useful information but incomplete or partially inaccurate)
- 7-8: Good (clear, mostly accurate, and relevant)
- 9-10: Excellent (precise, accurate, and highly relevant for diagnosis)

## Examples of Good Captions
1. "Apple leaf exhibiting Alternaria blotch. Small, circular brown lesions with yellowish halos are visible on the blade, some starting to coalesce near the margins. Spots appear slightly sunken and surrounded by chlorosis, indicating early to mid infection. In warm, humid conditions, lesions can expand, trigger premature defoliation, and reduce tree vigor and fruit quality."

2. "Tomato leaf affected by Spider Mites. Fine, pale stippling spreads across the surface, especially along veins, giving a yellow‑bronze, roughened appearance. Leaf edges curl and pucker, and tissue feels dry, with tiny specks and faint webbing likely on the underside. Feeding reduces chlorophyll and vigor, leading to scorch and premature leaf drop in hot, dry weather."

{format_instructions}
"""
evaluation_system_message_prompt = SystemMessagePromptTemplate.from_template(evaluation_system_template)

# ========== Evaluation Human Prompt ==========
evaluation_human_template = "Please evaluate the following image caption:\n\n{caption_text}"
evaluation_human_message_prompt = HumanMessagePromptTemplate.from_template(evaluation_human_template)

# ========== Create Evaluation Prompt Template ==========
evaluation_prompt = ChatPromptTemplate.from_messages([
    evaluation_system_message_prompt,
    evaluation_human_message_prompt
])

# ========== Optimization System Prompt ==========
optimization_system_template = """You are an expert agricultural diagnostician. Please optimize the following image caption to make it more accurate, detailed, and professional.

## Optimization Guidelines
1. Ensure clear identification of plant and disease/pest (or confirm health status)
2. Include detailed symptom description (location, shape, color, extent, quantity, etc.)
3. Assess severity and development stage
4. Keep language concise and professional (80-120 words)
5. Follow the style and quality of the examples below

## Examples of Excellent Captions
1. "Apple leaf exhibiting Alternaria blotch. Small, circular brown lesions with yellowish halos are visible on the blade, some starting to coalesce near the margins. Spots appear slightly sunken and surrounded by chlorosis, indicating early to mid infection. In warm, humid conditions, lesions can expand, trigger premature defoliation, and reduce tree vigor and fruit quality."

2. "Tomato leaf affected by Spider Mites. Fine, pale stippling spreads across the surface, especially along veins, giving a yellow‑bronze, roughened appearance. Leaf edges curl and pucker, and tissue feels dry, with tiny specks and faint webbing likely on the underside. Feeding reduces chlorophyll and vigor, leading to scorch and premature leaf drop in hot, dry weather."

Please provide an optimized version of the caption:
"""
optimization_system_message_prompt = SystemMessagePromptTemplate.from_template(optimization_system_template)

# ========== Optimization Human Prompt ==========
optimization_human_template = "Original caption: {caption_text}\n\nSuggestions for improvement: {suggestions}"
optimization_human_message_prompt = HumanMessagePromptTemplate.from_template(optimization_human_template)

# ========== Create Optimization Prompt Template ==========
optimization_prompt = ChatPromptTemplate.from_messages([
    optimization_system_message_prompt,
    optimization_human_message_prompt
])

# ========== Initialize Model ==========
model = ChatOpenAI(
    model="YOUR_MODEL_NAME",  # e.g., "gpt-4", "gpt-3.5-turbo", etc.
    temperature=0,
    max_retries=3,
    timeout=30,
)


# ========== JSON Repair Function ==========
def extract_and_fix_json(text):
    """Extract and fix JSON format from text"""
    if not isinstance(text, str) or not text.strip():
        return {"rating": 0, "reasoning": "No valid response", "suggestions": "No valid response"}

    text = text.strip()

    # First try direct parsing
    try:
        result = json.loads(text)
        if isinstance(result, dict) and "rating" in result:
            return {
                "rating": result.get("rating", 0),
                "reasoning": str(result.get("reasoning", "")),
                "suggestions": str(result.get("suggestions", ""))
            }
    except:
        pass

    # Try to find JSON object
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx >= 0 and end_idx > start_idx:
        json_str = text[start_idx:end_idx + 1]
        try:
            # Quickly fix common JSON format errors
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            json_str = re.sub(r'([{,])\s*([^"{}\[\]]+?)\s*:', r'\1"\2":', json_str)

            result = json.loads(json_str)
            if isinstance(result, dict):
                return {
                    "rating": result.get("rating", 0),
                    "reasoning": str(result.get("reasoning", "")),
                    "suggestions": str(result.get("suggestions", ""))
                }
        except:
            pass

    # If all attempts fail, return default values
    return {"rating": 0, "reasoning": "Failed to parse response", "suggestions": "Check the caption format"}


# ========== Retry Decorator for API Calls ==========
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_model_with_retry(model, messages):
    """Call the model with retry mechanism"""
    return model.invoke(messages)


# ========== Evaluate Caption ==========
def evaluate_caption(caption_text):
    """Evaluate the quality of a caption"""
    try:
        # Format the evaluation prompt
        messages = evaluation_prompt.format_messages(
            caption_text=caption_text,
            format_instructions=evaluation_format_instructions
        )

        # Call the model
        response = call_model_with_retry(model, messages)

        # Parse the response
        try:
            parsed = evaluation_parser.parse(response.content)
            return {
                "rating": int(parsed.get("rating", 0)),
                "reasoning": parsed.get("reasoning", ""),
                "suggestions": parsed.get("suggestions", "")
            }
        except Exception as e:
            # If standard parsing fails, use the repair function
            print(f"Evaluation parsing failed: {e}")
            return extract_and_fix_json(response.content)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {"rating": 0, "reasoning": f"Evaluation error: {str(e)}", "suggestions": "Try again"}


# ========== Optimize Caption ==========
def optimize_caption(caption_text, suggestions):
    """Optimize a caption based on suggestions"""
    try:
        # Format the optimization prompt
        messages = optimization_prompt.format_messages(
            caption_text=caption_text,
            suggestions=suggestions
        )

        # Call the model
        response = call_model_with_retry(model, messages)

        # Return the optimized caption
        return response.content.strip()

    except Exception as e:
        print(f"Optimization failed: {e}")
        return caption_text  # Return original caption if optimization fails


# ========== Load and Save Functions ==========
def load_image_captions(file_path):
    """Load image captions from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_image_captions(file_path, captions):
    """Save image captions to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(captions, file, indent=4, ensure_ascii=False)


# ========== Process and Optimize Captions ==========
def process_and_optimize_captions(captions, threshold=8):
    """Process and optimize low-scoring image captions"""
    for i, caption in enumerate(tqdm(captions, desc="Evaluating and optimizing captions")):
        caption_text = caption.get("image_caption", "")

        # Skip if no caption or already processed
        if not caption_text or caption.get("evaluated", False):
            continue

        # Evaluate the caption
        evaluation = evaluate_caption(caption_text)

        # Add evaluation results to the caption
        captions[i]["rating"] = evaluation["rating"]
        captions[i]["reasoning"] = evaluation["reasoning"]
        captions[i]["suggestions"] = evaluation["suggestions"]
        captions[i]["evaluated"] = True

        # Store original caption before optimization
        if "original_caption" not in captions[i]:
            captions[i]["original_caption"] = caption_text

        # Optimize if rating is below threshold
        if evaluation["rating"] < threshold:
            optimized_caption = optimize_caption(caption_text, evaluation["suggestions"])
            captions[i]["image_caption"] = optimized_caption  # Replace with optimized version
            captions[i]["optimized"] = True

            # Print progress
            print(f"\nOptimized caption {i + 1}:")
            print(f"  Original: {caption_text[:80]}...")
            print(f"  Optimized: {optimized_caption[:80]}...")
            print(f"  Rating: {evaluation['rating']}/10")
        else:
            captions[i]["optimized"] = False

    return captions


# ========== Main Function ==========
def main():
    parser = argparse.ArgumentParser(description='Caption Judge and Optimize Script')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file path')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file path')
    parser.add_argument('--threshold', '-t', type=int, default=8,
                       help='Quality threshold (1-10). Captions below this will be optimized. Default: 8')

    args = parser.parse_args()

    # Load image captions
    print(f"Loading captions from {args.input}...")
    image_captions = load_image_captions(args.input)

    # Process and optimize captions
    print(f"Evaluating and optimizing captions (threshold: {args.threshold})...")
    updated_captions = process_and_optimize_captions(image_captions, args.threshold)

    # Save the updated captions
    print(f"Saving results to {args.output}...")
    save_image_captions(args.output, updated_captions)

    # Calculate statistics
    total_count = len(updated_captions)
    evaluated_count = sum(1 for c in updated_captions if c.get("evaluated", False))
    optimized_count = sum(1 for c in updated_captions if c.get("optimized", False))

    print(f"\nProcessing complete!")
    print(f"Total captions: {total_count}")
    print(f"Evaluated captions: {evaluated_count}")
    print(f"Optimized captions: {optimized_count} ({optimized_count/total_count*100:.1f}%)")

    if evaluated_count > 0:
        avg_rating = sum(c.get("rating", 0) for c in updated_captions if c.get("evaluated", False)) / evaluated_count
        print(f"Average rating: {avg_rating:.2f}/10")


if __name__ == "__main__":
    main()

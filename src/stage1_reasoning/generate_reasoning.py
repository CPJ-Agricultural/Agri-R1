# coding: utf-8
import os
import base64
import json
import time
import re
import requests
from collections import OrderedDict
from tenacity import retry, stop_after_attempt, wait_exponential

# ========== Configuration ==========
API_BASE = "***"
API_KEY = "***"

# Input and output JSON files
input_json = "shiyanmethods.json"
output_json = "shiyanmethods-03.json"

# Image base directory
BASE_IMAGE_DIR = "dataset01"

TOTAL_COT_TARGET = 200000
SAVE_INTERVAL = 10000

# ========== Enhanced System Prompt ==========
system_prompt = """You are a plant disease management expert. Generate clear, step-by-step reasoning for agricultural questions.

## Core Requirements:
1. Output must be in English and structured into 3-4 explicit steps labeled "Step 1: … Step 2: …"
2. For PREVENTION/CONTROL questions: Focus ONLY on method reasoning - DO NOT re-diagnose
3. For IDENTIFICATION questions: Focus on visual evidence and diagnostic reasoning
4. Keep reasoning concise and practical

## Question Type Guidelines:

### FOR PREVENTION/CONTROL METHODS QUESTIONS:
- Step 1: Analyze disease characteristics that influence control strategies
- Step 2: Recommend cultural/preventive practices based on disease biology
- Step 3: Outline chemical control timing and selection
- Step 4: Integrate application methods and safety precautions

### FOR DISEASE IDENTIFICATION QUESTIONS:
- Step 1: Plant identification based on morphological features
- Step 2: Symptom observation and description  
- Step 3: Disease pattern analysis
- Step 4: Preliminary diagnosis with confidence level

## CRITICAL RULES:
- If question asks about CONTROL/MANAGEMENT/PREVENTION/TREATMENT/METHODS: Use PREVENTION/CONTROL guideline
- If question asks about IDENTIFICATION/WHAT/NAME: Use IDENTIFICATION guideline
- NEVER mix guidelines - choose one based on question type

## Output Format:
{"think": "Step 1: … Step 2: … Step 3: …"}

Output only the JSON string.
"""

# ========== Optimized Few-shot Examples ==========
examples = {
    "prevention_control": {
        "question": "What effective prevention and control methods can be applied to Apple Brown Spot?",
        "answer": "(1) Prune appropriately, improve drainage, and enhance ventilation and light penetration in the orchard.\n(2) In fall and winter, clean up fallen leaves and diseased branches and leaves in the orchard, and bury or burn them deeply.\n(3) Apply protective fungicides starting in mid-May, every 15 days, for a total of 3-4 applications. Common fungicides include Bordeaux mixture (1:2:200), 30% Captan 500x solution, 77% Kocide 800x solution, 70% Thiophanate-methyl 800x solution, 70% Mancozeb 500x solution, and 75% Chlorothalonil 800x solution. Note that using Bordeaux mixture during the young fruit stage can cause fruit russeting.",
        "think": "Step 1: Based on fungal disease characteristics, recommend pruning for improved air circulation to reduce humidity. Step 2: Advise sanitation measures in fall/winter to eliminate overwintering fungal spores in plant debris. Step 3: Schedule fungicide applications from mid-May to target primary infection periods during the growing season. Step 4: Select appropriate fungicide combinations and note application precautions for comprehensive disease management."
    },
    "identification": {
        "question": "What is the content of this picture?",
        "answer": "This image shows an apple leaf affected by Alternaria Blotch.",
        "think": "Step 1: Identify plant - leaf ovate with serrated margin and pinnate venation, consistent with apple leaf morphology. Step 2: Describe symptoms - multiple circular brown lesions with yellowish halos scattered across leaf surface. Step 3: Assess distribution - lesions measure approximately 2-5 mm in diameter and cover about 20% of visible leaf area. Step 4: Preliminary diagnosis - Alternaria Blotch based on characteristic lesion appearance; confidence: medium."
    },
    "verification": {
        "question": "Is this leaf from a pear tree?",
        "answer": "No, this is an apple leaf.",
        "think": "Step 1: Analyze leaf shape - ovate form with pointed tip differs from pear's broader shape. Step 2: Examine margins - fine serrations present, unlike pear's finer serration pattern. Step 3: Study venation - pinnate pattern with 45-degree branching typical of apple. Step 4: Confirm identification - all morphological features consistent with apple leaf; confidence: high."
    }
}


# ========== Question Type Detection ==========
def detect_question_type(question):
    """Detect if question is about prevention/control or identification"""
    question_lower = question.lower()

    # Keywords for prevention/control questions
    control_keywords = [
        'control', 'prevention', 'management', 'treatment','methods', 'method',
        'how to', 'what measures', 'how can', 'ways to', 'strategies',
        'protect', 'avoid', 'reduce', 'eliminate', 'combat', 'fight'
    ]

    # Keywords for identification questions
    identification_keywords = [
        'what is', 'identify', 'name', 'diagnose', 'what disease',
        'what kind', 'what type', 'what are the symptoms'
    ]

    for keyword in control_keywords:
        if keyword in question_lower:
            return "prevention_control"

    for keyword in identification_keywords:
        if keyword in question_lower:
            return "identification"

    # Default to prevention_control for ambiguous cases (based on your use case)
    return "prevention_control"


# ========== Optimized Build Prompt ==========
def build_prompt(original_question, original_answer):
    """Build prompt with question type specific examples"""

    question_type = detect_question_type(original_question)

    # Select primary example based on question type
    if question_type == "prevention_control":
        primary_example = examples["prevention_control"]
        secondary_examples = [examples["identification"], examples["verification"]]
        task_instruction = "THIS IS A PREVENTION/CONTROL QUESTION - FOCUS ON METHODS ONLY, DO NOT RE-DIAGNOSE"
    else:
        primary_example = examples["identification"]
        secondary_examples = [examples["verification"], examples["prevention_control"]]
        task_instruction = "This is an identification question - focus on visual evidence and diagnosis"

    prompt = f"""{system_prompt}

## PRIMARY EXAMPLE ({question_type.upper().replace('_', ' ')}):
Question: {primary_example['question']}
Answer: {primary_example['answer']}
Think: {primary_example['think']}

## ADDITIONAL EXAMPLES:
Question: {secondary_examples[0]['question']}
Answer: {secondary_examples[0]['answer']}
Think: {secondary_examples[0]['think']}

Question: {secondary_examples[1]['question']}
Answer: {secondary_examples[1]['answer']}
Think: {secondary_examples[1]['think']}

## CURRENT TASK:
{task_instruction}

Question: {original_question}
Answer: {original_answer}

Generate thinking chain following the appropriate guideline above.

Return only JSON format:"""
    return prompt


# ========== API Call Function (保持不变) ==========
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_siliconflow_api(image_base64, prompt_text):
    """Direct call to SiliconFlow API"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-ai/DeepSeek-VL2",
        "messages": [
            {
                "role": "user",
                "content": [

                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }

    try:
        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"API call failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
        raise


# ========== JSON Repair Function (保持不变) ==========
def extract_and_fix_json(text):
    """Extract and fix JSON format"""
    if not isinstance(text, str) or not text.strip():
        return {"think": "No valid response"}

    text = text.strip()

    # Try direct parsing first
    try:
        result = json.loads(text)
        if isinstance(result, dict) and "think" in result:
            return {"think": str(result.get("think", ""))}
    except:
        pass

    # Try to find JSON object
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx >= 0 and end_idx > start_idx:
        json_str = text[start_idx:end_idx + 1]
        try:
            # Fix common JSON format errors
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            json_str = re.sub(r'([{,])\s*([^"{}\[\]]+?)\s*:', r'\1"\2":', json_str)

            result = json.loads(json_str)
            if isinstance(result, dict):
                return {"think": str(result.get("think", ""))}
        except:
            pass

    # Try to extract step format
    steps = re.findall(r'Step \d+:.*?(?=Step \d+|$)', text, re.IGNORECASE | re.DOTALL)
    if steps:
        return {"think": " ".join(step.strip() for step in steps)}

    return {"think": text[:500]}


# ========== Enhanced Processing Function ==========
def process_conversation_item(entry, idx, total, total_cot_count):
    """Process single conversation item with enhanced question type handling"""
    if "image" not in entry or "conversations" not in entry:
        print(f"❌ [{idx}/{total}] Missing required fields, skipping")
        return entry, 0

    # Process image path
    image_path = entry["image"]
    if not os.path.exists(image_path):
        image_path = os.path.join(BASE_IMAGE_DIR, os.path.basename(image_path))

    conversations = entry["conversations"]

    # Read image and convert to base64
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        print(f"  📷 Successfully read image: {os.path.basename(image_path)}")
    except Exception as e:
        print(f"❌ [{idx}/{total}] Failed to read image {image_path}: {e}")
        for conv in conversations:
            if conv["from"] == "gpt":
                conv["think"] = f"Image read failed: {str(e)}"
        return entry, 0

    # Process each GPT response
    processed_count = 0
    for i, conv in enumerate(conversations):
        if total_cot_count + processed_count >= TOTAL_COT_TARGET:
            print(f"🎯 Reached target of {TOTAL_COT_TARGET} Reasoning chains, stopping processing")
            break

        if conv["from"] == "gpt":
            # Find corresponding human question
            if i > 0 and conversations[i - 1]["from"] == "human":
                human_question = conversations[i - 1]["value"]
                human_question = human_question.replace("<image>\n", "").replace("<image>", "").strip()
                gpt_answer = conv["value"]

                question_type = detect_question_type(human_question)

                print(f"  🔍 Processing Q-A {i // 2 + 1} ({question_type}): Q: {human_question[:50]}...")

                # Build prompt
                prompt = build_prompt(human_question, gpt_answer)

                # Call API to generate thinking chain
                try:
                    response_content = call_siliconflow_api(image_data, prompt)
                    cot_result = extract_and_fix_json(response_content)
                    conv["think"] = cot_result["think"]
                    processed_count += 1

                    # Validate thinking chain format
                    think_text = cot_result["think"]
                    step_count = len(re.findall(r'Step \d+:', think_text))
                    print(f"  ✅ Generated {step_count}-step {question_type} thinking chain")

                    # Debug: Print first 100 chars of thinking chain
                    if step_count == 0:
                        print(f"  ⚠️  Warning: No steps detected in thinking chain: {think_text[:100]}...")

                except Exception as e:
                    error_msg = f"Reasoning generation failed: {str(e)}"
                    print(f"  ❌ Failed to generate thinking chain: {e}")
                    conv["think"] = error_msg
            else:
                conv["think"] = "No corresponding human question found"

    print(f"✅ [{idx}/{total}] Processed {os.path.basename(image_path)} - {processed_count} thinking chains")
    return entry, processed_count


# ========== Main Processing (保持不变) ==========
def main():
    try:
        with open(input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"📁 Successfully read input file, total {len(data) if isinstance(data, list) else 1} items")
    except Exception as e:
        print(f"❌ Failed to read input JSON: {e}")
        return

    if isinstance(data, dict):
        data = [data]

    results = []
    total = len(data)
    processed_count = 0
    total_cot_count = 0
    start_time = time.time()

    print(f"🚀 Starting to process {total} items...")
    print(f"🎯 Target: Generate {TOTAL_COT_TARGET} Reasoning chains")

    for idx, entry in enumerate(data, start=1):
        if total_cot_count >= TOTAL_COT_TARGET:
            print(f"🎯 Reached target of {TOTAL_COT_TARGET} Reasoning chains, stopping processing")
            break

        print(f"\n📝 Processing item {idx}/{total}...")
        processed_entry, new_cot_count = process_conversation_item(entry, idx, total, total_cot_count)
        results.append(processed_entry)
        total_cot_count += new_cot_count

        # 统计成功处理的条目
        has_valid_cot = False
        valid_cot_count = 0
        if "conversations" in processed_entry:
            for conv in processed_entry["conversations"]:
                if (conv.get("from") == "gpt" and conv.get("think") and
                        not conv["think"].startswith(
                            ("Reasoning generation failed", "Image read failed", "No corresponding"))):
                    has_valid_cot = True
                    valid_cot_count += 1

        if has_valid_cot:
            processed_count += 1

        # 定期保存
        if total_cot_count > 0 and total_cot_count % SAVE_INTERVAL == 0:
            checkpoint_file = f"checkpoint_{total_cot_count}_{output_json}"
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Checkpoint saved: {checkpoint_file} with {total_cot_count} Reasoning chains")

        if idx % 2 == 0:
            with open(f"temp_{output_json}", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    # 保存最终结果
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 计算统计信息
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_item = total_time / processed_count if processed_count > 0 else 0
    avg_time_per_cot = total_time / total_cot_count if total_cot_count > 0 else 0

    print(f"\n🎉 Successfully generated {output_json}")
    print(f"📊 Successfully processed {processed_count}/{total} items")
    print(f"🔗 Total {total_cot_count} thinking chains generated")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"📈 Average per item: {avg_time_per_item:.2f} seconds")
    print(f"📈 Average per Reasoning chain: {avg_time_per_cot:.2f} seconds")

    if total_cot_count >= TOTAL_COT_TARGET:
        print(f"\n🎯 TARGET REACHED: Successfully generated {TOTAL_COT_TARGET} Reasoning chains!")


if __name__ == "__main__":
    main()
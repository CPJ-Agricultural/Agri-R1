import os
import json
import argparse
from copy import deepcopy
from typing import List
from pathlib import Path
from tqdm import tqdm
from langchain_openai import ChatOpenAI


os.environ["OPENAI_API_BASE"] = "****"
os.environ["OPENAI_API_KEY"] = "****"


INSTRUCT_PROMPT = """We would like to request your feedback on the performance of two AI assistants in response to the crop disease knowledge question displayed above. The user asks the question on observing a crop disease image.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""


def conv_to_str(image_path, question, reference_ans, model_ans):
    """Generate instruction for GPT to score the two answers."""
    return (f'[Context]\n'
            f'Image: {image_path}\n'
            f'This is a crop disease image used for knowledge VQA.\n\n'
            f'[Question]\n{question}\n\n'
            f'[Assistant 1 - Reference Answer]\n{reference_ans}\n\n[End of Assistant 1]\n\n'
            f'[Assistant 2 - Model Answer]\n{model_ans}\n\n[End of Assistant 2]\n\n'
            f'[System]\n{INSTRUCT_PROMPT}\n\n')


def compare_messages_gen(image_path, question, reference_ans, model_ans):
    """Generate messages for GPT evaluation."""
    messages = [
        {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
    ]
    messages.append({
        "role": "user",
        "content": conv_to_str(image_path, question, reference_ans, model_ans)
    })
    return messages


def chunk(lst, n):
    """Split list into chunks of size n."""
    for i in range(0, len(lst), n):
        if i + (1.5 * n) < len(lst):
            end = i + n
        else:
            end = len(lst)
        yield lst[i:end]
        if end == len(lst):
            return


def evaluate_with_gpt(samples, model_name="gpt-4"):
    """
    Evaluate samples using GPT model.

    Args:
        samples: List of samples with 'question_id', 'image', 'question', 'reference_answer', 'model_answer'
        model_name: GPT model name to use for evaluation

    Returns:
        List of results with GPT evaluation scores
    """
    # Initialize LangChain ChatOpenAI model
    model = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_retries=3,
        timeout=30,
    )

    BATCH_SIZE = 1
    results = []

    print(f'Starting Crop Disease Knowledge GPT Scoring Evaluation')
    print(f'Total samples: {len(samples)}')
    print(f'Using model: {model_name}')

    for sample in tqdm(samples, desc="Evaluating"):
        try:
            # Generate comparison prompt
            messages = compare_messages_gen(
                sample['image'],
                sample['question'],
                sample['reference_answer'],
                sample['model_answer']
            )

            # Convert to LangChain format
            from langchain_core.messages import HumanMessage, SystemMessage
            lc_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    lc_messages.append(SystemMessage(content=msg['content']))
                elif msg['role'] == 'user':
                    lc_messages.append(HumanMessage(content=msg['content']))

            # Get GPT evaluation
            response = model.invoke(lc_messages)
            gpt_eval = response.content.strip()

            # Parse score from response
            score_line = gpt_eval.split('\n')[0].strip()
            scores = score_line.split()

            result = deepcopy(sample)
            result['gpt_eval'] = gpt_eval
            result['gpt_eval_raw'] = score_line

            # Extract scores if available
            if len(scores) >= 2:
                try:
                    result['reference_score'] = float(scores[0])
                    result['model_score'] = float(scores[1])
                except ValueError:
                    result['reference_score'] = None
                    result['model_score'] = None

            results.append(result)

        except Exception as e:
            print(f"\nError evaluating question_id {sample.get('question_id', 'unknown')}: {e}")
            result = deepcopy(sample)
            result['gpt_eval'] = f"Error: {str(e)}"
            result['reference_score'] = None
            result['model_score'] = None
            results.append(result)

    print(f"\nEvaluation completed. Result size: {len(results)}")
    return results


def calculate_statistics(results):
    """Calculate and print evaluation statistics."""
    valid_results = [r for r in results if r.get('model_score') is not None]

    if not valid_results:
        print("No valid scores found!")
        return

    model_scores = [r['model_score'] for r in valid_results]
    reference_scores = [r['reference_score'] for r in valid_results]

    avg_model_score = sum(model_scores) / len(model_scores)
    avg_reference_score = sum(reference_scores) / len(reference_scores)

    # Calculate total scores
    total_model_score = sum(model_scores)
    total_reference_score = sum(reference_scores)

    # Convert to 100-point scale
    # Maximum possible score = number_of_questions * 10
    max_possible_score = len(valid_results) * 10
    model_score_100 = (total_model_score / max_possible_score) * 100
    reference_score_100 = (total_reference_score / max_possible_score) * 100

    print("\n" + "="*50)
    print("Evaluation Statistics")
    print("="*50)
    print(f"Total samples: {len(results)}")
    print(f"Valid evaluations: {len(valid_results)}")
    print(f"\n--- Per Question Scores ---")
    print(f"Average Model Score: {avg_model_score:.2f} / 10")
    print(f"Average Reference Score: {avg_reference_score:.2f} / 10")
    print(f"\n--- Total Scores ---")
    print(f"Total Model Score: {total_model_score:.2f} / {max_possible_score}")
    print(f"Total Reference Score: {total_reference_score:.2f} / {max_possible_score}")
    print(f"\n--- 100-Point Scale (Normalized) ---")
    print(f"Model Score (100-point): {model_score_100:.2f} / 100")
    print(f"Reference Score (100-point): {reference_score_100:.2f} / 100")
    print(f"\n--- Relative Performance ---")
    print(f"Relative Performance: {(avg_model_score/avg_reference_score)*100:.2f}%")
    print("="*50)


def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main(args):
    # Load ground truth data (with reference answers)
    print(f"Loading ground truth from: {args.ground_truth_file}")
    if args.ground_truth_file.endswith('.jsonl'):
        ground_truth_data = load_jsonl(args.ground_truth_file)
    else:
        ground_truth_data = load_json(args.ground_truth_file)

    # Load model predictions
    print(f"Loading model predictions from: {args.model_answers_file}")
    if args.model_answers_file.endswith('.jsonl'):
        model_answers_data = load_jsonl(args.model_answers_file)
    else:
        model_answers_data = load_json(args.model_answers_file)

    # Create mapping for model answers
    model_answers_dict = {item['question_id']: item for item in model_answers_data}

    # Prepare samples for evaluation
    samples = []
    for gt_item in ground_truth_data:
        question_id = gt_item['question_id']

        if question_id not in model_answers_dict:
            print(f"Warning: No model answer found for question_id: {question_id}")
            continue

        model_item = model_answers_dict[question_id]

        sample = {
            'question_id': question_id,
            'image': gt_item['image'],
            'question': gt_item['question'],
            'reference_answer': gt_item['answer'],  # Ground truth answer
            'model_answer': model_item.get('solution', model_item.get('answer', model_item.get('prediction', '')))  # Model's answer
        }
        samples.append(sample)

    print(f"\nPrepared {len(samples)} samples for evaluation")

    if len(samples) == 0:
        print("Error: No samples to evaluate!")
        return

    # Evaluate using GPT
    results = evaluate_with_gpt(samples, model_name=args.model_name)

    # Create parent directory of output score files if it doesn't exist
    os.makedirs(Path(args.output_file).parent, exist_ok=True)

    # Save results
    print(f"\nSaving results to: {args.output_file}")
    if args.output_file.endswith('.jsonl'):
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    else:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Calculate and print statistics
    calculate_statistics(results)

    print(f"\nEvaluation completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Crop Disease Knowledge GPT Evaluation", add_help=True)
    parser.add_argument(
        "--ground-truth-file",
        required=True,
        metavar="FILE",
        help="Path to ground truth file (JSON or JSONL) containing question_id, image, question, answer"
    )
    parser.add_argument(
        "--model-answers-file",
        required=True,
        metavar="FILE",
        help="Path to model predictions file (JSON or JSONL) containing question_id and answer/prediction"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        metavar="FILE",
        help="Path to save evaluation results (JSON or JSONL)"
    )
    parser.add_argument(
        "--model-name",
        default="gpt-5-mini",
        help="GPT model name for evaluation (default: gpt-5-mini)"
    )

    args = parser.parse_args()
    main(args)

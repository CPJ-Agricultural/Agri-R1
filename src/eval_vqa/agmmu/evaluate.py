"""
AgMMU Evaluation Script
Supports MCQ accuracy evaluation and OEQ LLM-as-judge evaluation
"""

import json
import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

class AgMMUEvaluator:
    """AgMMU Dataset Evaluator"""

    def __init__(self, use_llm_judge: bool = True):
        self.use_llm_judge = use_llm_judge
        self.results = {
            'mcq': defaultdict(list),
            'oeq': defaultdict(list)
        }

    def evaluate_mcq(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Evaluate MCQ accuracy

        Args:
            predictions: Model prediction results [{"id": "xxx", "prediction": "A"}, ...]
            ground_truth: Ground truth answers [{"id": "xxx", "answer": "correct answer", "options": [...]}]

        Returns:
            Evaluation results dictionary
        """
        # Build ID to answer mapping
        gt_dict = {item['id']: item for item in ground_truth}
        pred_dict = {item['id']: item['prediction'] for item in predictions}

        results_by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
        overall_results = {'correct': 0, 'total': 0}

        for item_id, gt_item in gt_dict.items():
            if item_id not in pred_dict:
                continue

            qtype = gt_item['qtype']
            prediction = pred_dict[item_id].strip().upper()

            # Extract predicted option letter (A/B/C/D)
            predicted_letter = self._extract_choice_letter(prediction)

            # Find the correct option letter corresponding to the answer
            correct_letter = self._find_correct_letter(
                gt_item['answer'],
                gt_item['options']
            )

            # Check if correct
            is_correct = (predicted_letter == correct_letter)

            # Statistics
            results_by_type[qtype]['correct'] += int(is_correct)
            results_by_type[qtype]['total'] += 1
            overall_results['correct'] += int(is_correct)
            overall_results['total'] += 1

        # Calculate accuracy
        accuracy_by_type = {}
        for qtype, stats in results_by_type.items():
            accuracy_by_type[qtype] = {
                'accuracy': stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total']
            }

        overall_accuracy = overall_results['correct'] / overall_results['total'] * 100

        return {
            'overall_accuracy': overall_accuracy,
            'overall_correct': overall_results['correct'],
            'overall_total': overall_results['total'],
            'by_type': accuracy_by_type
        }

    def evaluate_oeq(self, predictions: List[Dict], ground_truth: List[Dict],
                     llm_judge_model=None) -> Dict:
        """
        Evaluate OEQ (Open-Ended Questions)

        Uses AgMMU paper's LLM-as-judge method
        Scoring: CORRECT / PARTIALLY_CORRECT / INCORRECT / NOT_ATTEMPTED
        """
        if not self.use_llm_judge:
            # Simple string matching
            return self._evaluate_oeq_simple(predictions, ground_truth)
        else:
            # Use LLM as judge
            return self._evaluate_oeq_llm_judge(predictions, ground_truth, llm_judge_model)

    def _extract_choice_letter(self, text: str) -> str:
        """Extract option letter A/B/C/D from text"""
        # Match first occurrence of A/B/C/D
        match = re.search(r'\b([A-D])\b', text.upper())
        if match:
            return match.group(1)
        return ""

    def _find_correct_letter(self, answer: str, options: List[str]) -> str:
        """Find option letter corresponding to answer text"""
        answer_lower = answer.lower().strip()

        for i, option in enumerate(options):
            option_lower = option.lower().strip()
            if answer_lower == option_lower or answer_lower in option_lower:
                return chr(65 + i)  # A=65

        return ""

    def _evaluate_oeq_simple(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Simple OEQ evaluation (string matching)"""
        gt_dict = {item['id']: item for item in ground_truth}
        pred_dict = {item['id']: item['prediction'] for item in predictions}

        results_by_type = defaultdict(lambda: {'correct': 0, 'total': 0})

        for item_id, gt_item in gt_dict.items():
            if item_id not in pred_dict:
                continue

            qtype = gt_item['qtype']
            prediction = pred_dict[item_id].lower().strip()
            answer = gt_item['answer'].lower().strip()

            # Simple containment matching
            is_correct = answer in prediction or prediction in answer

            results_by_type[qtype]['correct'] += int(is_correct)
            results_by_type[qtype]['total'] += 1

        # Calculate accuracy
        accuracy_by_type = {}
        for qtype, stats in results_by_type.items():
            accuracy_by_type[qtype] = {
                'accuracy': stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total']
            }

        return {'by_type': accuracy_by_type}

    def _evaluate_oeq_llm_judge(self, predictions, ground_truth, llm_model):
        """
        Use LLM as judge to evaluate OEQ

        Reference AgMMU paper scoring standards:
        - CORRECT (1.0)
        - PARTIALLY_CORRECT (0.5)
        - INCORRECT (0.0)
        - NOT_ATTEMPTED (0.0)

        Final score = harmonic mean
        """
        # TODO: Implement LLM-as-judge
        # Need to call Qwen-7B or other LLM for scoring
        print("⚠️ LLM-as-judge evaluation requires LLM model configuration")
        return {}

    def print_results(self, mcq_results: Dict, oeq_results: Dict = None):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("📊 AgMMU Evaluation Results")
        print("="*60)

        # MCQ results
        print(f"\n🎯 Multiple Choice Questions (MCQ)")
        print(f"   Overall Accuracy: {mcq_results['overall_accuracy']:.2f}%")
        print(f"   ({mcq_results['overall_correct']}/{mcq_results['overall_total']})\n")

        print("   By knowledge type:")
        for qtype, stats in sorted(mcq_results['by_type'].items(),
                                   key=lambda x: x[1]['accuracy'], reverse=True):
            acc = stats['accuracy']
            correct = stats['correct']
            total = stats['total']
            print(f"   • {qtype:30s}: {acc:6.2f}% ({correct}/{total})")

        # OEQ results
        if oeq_results:
            print(f"\n📝 Open-Ended Questions (OEQ)")
            if 'by_type' in oeq_results:
                print("   By knowledge type:")
                for qtype, stats in sorted(oeq_results['by_type'].items(),
                                           key=lambda x: x[1]['accuracy'], reverse=True):
                    acc = stats['accuracy']
                    print(f"   • {qtype:30s}: {acc:6.2f}%")

        print("\n" + "="*60)


def compare_with_baselines(your_results: Dict, baseline_results: Dict):
    """
    Compare with baselines in AgMMU paper

    Baselines:
    - GPT-4o: MCQ 85.25%, OEQ 17.79%
    - LLaVA-1.5-7B: MCQ 64.16%, OEQ 10.42%
    """
    print("\n" + "="*60)
    print("📈 Comparison with AgMMU Baselines")
    print("="*60)

    baselines = {
        'GPT-4o': {'mcq': 85.25, 'oeq': 17.79},
        'Gemini 1.5 Pro': {'mcq': 80.42, 'oeq': 21.98},
        'LLaVA-1.5-7B': {'mcq': 64.16, 'oeq': 10.42},
        'LLaVA-1.5-13B': {'mcq': 66.73, 'oeq': 11.68}
    }

    your_mcq = your_results.get('mcq_accuracy', 0)
    your_oeq = your_results.get('oeq_accuracy', 0)

    print(f"\n{'Model':25s} {'MCQ':>10s} {'OEQ':>10s}")
    print("-" * 60)

    # Your model
    print(f"{'Your Model':25s} {your_mcq:9.2f}% {your_oeq:9.2f}%")
    print("-" * 60)

    # Baselines
    for model_name, scores in baselines.items():
        mcq_score = scores['mcq']
        oeq_score = scores['oeq']

        mcq_diff = your_mcq - mcq_score
        oeq_diff = your_oeq - oeq_score

        mcq_str = f"{mcq_score:.2f}%"
        oeq_str = f"{oeq_score:.2f}%"

        if mcq_diff > 0:
            mcq_str += f" (↑{mcq_diff:+.2f})"
        elif mcq_diff < 0:
            mcq_str += f" (↓{abs(mcq_diff):.2f})"

        if oeq_diff > 0:
            oeq_str += f" (↑{oeq_diff:+.2f})"
        elif oeq_diff < 0:
            oeq_str += f" (↓{abs(oeq_diff):.2f})"

        print(f"{model_name:25s} {mcq_str:>20s} {oeq_str:>20s}")

    print("="*60)


if __name__ == "__main__":
    # Example usage
    evaluator = AgMMUEvaluator(use_llm_judge=False)

    # Load ground truth
    with open("agmmu_qwen_format.json", 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    # Load model predictions (example)
    # predictions = [
    #     {"id": "agmmu_766423", "prediction": "B"},
    #     ...
    # ]

    # Evaluate MCQ
    # mcq_results = evaluator.evaluate_mcq(predictions, ground_truth)
    # evaluator.print_results(mcq_results)

    print("✅ Evaluation script ready")
    print("📝 Usage:")
    print("   1. Run model to get prediction results")
    print("   2. Call evaluator.evaluate_mcq() to evaluate MCQ")
    print("   3. Call evaluator.evaluate_oeq() to evaluate OEQ")

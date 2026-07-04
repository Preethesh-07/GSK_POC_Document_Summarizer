"""
ROUGE evaluation metric.

Evaluates summary quality using ROUGE-1, ROUGE-2, and ROUGE-L.
Requires `rouge-score` package.
"""

from rouge_score import rouge_scorer


class RougeEvaluator:
    """Evaluates text using the ROUGE metric."""

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def evaluate(self, reference: str, hypothesis: str) -> dict[str, dict[str, float]]:
        """Evaluate a hypothesis summary against a reference summary.

        Args:
            reference: The ground-truth reference summary.
            hypothesis: The generated summary to evaluate.

        Returns:
            Dictionary containing precision, recall, and f1 scores
            for rouge1, rouge2, and rougeL.
        """
        scores = self.scorer.score(reference, hypothesis)
        
        return {
            metric: {
                "precision": score.precision,
                "recall": score.recall,
                "f1": score.fmeasure,
            }
            for metric, score in scores.items()
        }

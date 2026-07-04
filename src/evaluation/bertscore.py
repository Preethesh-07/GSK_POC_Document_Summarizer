"""
BERTScore evaluation metric.

Evaluates summary quality using BERTScore (semantic similarity).
Requires `bert-score` package.
"""

from bert_score import score as bert_score


class BertScoreEvaluator:
    """Evaluates text semantic similarity using BERTScore."""

    def evaluate(
        self, references: list[str], hypotheses: list[str]
    ) -> dict[str, float]:
        """Evaluate multiple hypothesis summaries against references.

        Args:
            references: List of ground-truth reference summaries.
            hypotheses: List of generated summaries to evaluate.

        Returns:
            Dictionary containing mean precision, recall, and f1 scores.
        """
        if not references or not hypotheses:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False)
        
        return {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F1.mean().item()),
        }

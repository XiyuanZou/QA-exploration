from rouge_score import rouge_scorer
import bert_score

# --- ROUGE Score ---
def compute_rougescore(ref, pred):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return scores

# --- BERTScore ---
def compute_bertscore(ref, pred):
    P, R, F1 = bert_score.score([pred], [ref], lang="en", verbose=False)
    return {
        "precision": P[0].item(),
        "recall": R[0].item(),
        "f1": F1[0].item()
    }

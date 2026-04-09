import csv
import json
import math
import os

def _cohen_kappa_binary(labels_a, labels_b):
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p0 = agree / n if n else 0.0
    p_a_true = sum(1 for x in labels_a if x) / n if n else 0.0
    p_b_true = sum(1 for x in labels_b if x) / n if n else 0.0
    pe = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if math.isclose(1 - pe, 0.0):
        return 1.0
    return (p0 - pe) / (1 - pe)

def compute_table8(annotation_dir="annotation"):
    mapping = {
        "hallucination_label": ("hallucination_labels.csv", "annotator_A_label", "annotator_B_label"),
        "contradiction_label": ("contradiction_labels.csv", "annotator_A_label", "annotator_B_label"),
        "safe_escalation_correctness": ("escalation_labels.csv", "annotator_A_correct", "annotator_B_correct"),
    }
    result = {}
    for key, (filename, col_a, col_b) in mapping.items():
        path = os.path.join(annotation_dir, filename)
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        a = [r[col_a].strip().upper() == "TRUE" for r in rows]
        b = [r[col_b].strip().upper() == "TRUE" for r in rows]
        agreement = 100.0 * sum(1 for x, y in zip(a, b) if x == y) / len(a) if a else 0.0
        kappa = _cohen_kappa_binary(a, b) if a else 0.0
        result[key] = {"kappa": round(kappa, 2), "percent_agreement": round(agreement, 1)}
    return result

if __name__ == "__main__":
    print(json.dumps(compute_table8(), indent=2))

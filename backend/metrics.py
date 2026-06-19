from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> List[List[int]]:
    """Baris = label aktual, kolom = label prediksi."""
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def overall_accuracy(cm: List[List[int]]) -> float:
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(len(cm)))
    return (correct / total) if total else 0.0


def per_class_metrics(cm: List[List[int]], classes: Sequence[str]) -> List[Dict]:
    n = len(classes)
    out: List[Dict] = []
    for i in range(n):
        tp = cm[i][i]
        support = sum(cm[i])
        pred_pos = sum(cm[r][i] for r in range(n))
        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out.append(
            {
                "label": classes[i],
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
        )
    return out


def macro_f1(per_class: List[Dict]) -> float:
    if not per_class:
        return 0.0
    return round(sum(p["f1"] for p in per_class) / len(per_class), 4)


def build_evaluation(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
    loss: Optional[float] = None,
    fine_tuned: bool = False,
    evaluated_at: Optional[str] = None,
) -> Dict:
    num = len(classes)
    cm = confusion_matrix(y_true, y_pred, num)
    pcm = per_class_metrics(cm, classes)
    return {
        "accuracy": round(overall_accuracy(cm), 4),
        "loss": (round(float(loss), 4) if loss is not None else None),
        "macro_f1": macro_f1(pcm),
        "num_classes": num,
        "num_samples": len(y_true),
        "classes": list(classes),
        "confusion_matrix": cm,
        "per_class": pcm,
        "fine_tuned": fine_tuned,
        "evaluated_at": evaluated_at,
    }


def save_metrics(metrics: Dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def load_metrics(path) -> Dict:
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)

"""Các chỉ số đánh giá cho bài toán phân loại."""

from typing import List

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def accuracy(preds: List[int], targets: List[int]) -> float:
    """Trả về accuracy giữa dự đoán và nhãn."""
    return accuracy_score(targets, preds)


def f1_macro(preds: List[int], targets: List[int]) -> float:
    """Trả về F1-score trung bình theo macro."""
    return f1_score(targets, preds, average="macro")


def conf_matrix(preds: List[int], targets: List[int]):
    """Trả về ma trận nhầm lẫn (confusion matrix)."""
    return confusion_matrix(targets, preds)

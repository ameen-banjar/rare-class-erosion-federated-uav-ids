import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

from model import AttackFamilyMLP, get_state_dict_copy


def make_loader(X, y, batch_size, shuffle=True):
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


def local_train(model, X, y, device, epochs=1, batch_size=256, lr=1e-3, mu=0.0, global_state=None,
                 class_weights=None):
    """One client's local training step. mu>0 + global_state activates the
    FedProx proximal term; mu=0 is plain local SGD (used by FedAvg/uniform/
    local-only/centralized). class_weights (same tensor for every client,
    computed once from the training pool) prevents collapse onto the
    majority classes (Regular/DoS are ~85% of rows)."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    loader = make_loader(X, y, batch_size)
    global_params = [p.clone().detach() for p in model.parameters()] if mu > 0 else None

    last_loss = None
    for _ in range(epochs):
        total_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            if mu > 0:
                prox = sum((p - gp).pow(2).sum() for p, gp in zip(model.parameters(), global_params))
                loss = loss + (mu / 2) * prox
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        last_loss = total_loss / max(n_batches, 1)
    return get_state_dict_copy(model), len(X), last_loss


def aggregate_weighted(client_states, weights):
    """weights: list summing to 1.0 (row-weighted for FedAvg/FedProx, uniform for client-uniform)."""
    agg = copy.deepcopy(client_states[0])
    for key in agg:
        agg[key] = sum(w * cs[key].float() for w, cs in zip(weights, client_states))
    return agg


@torch.no_grad()
def evaluate(model, X, y, device, class_names, batch_size=4096, class_weights=None,
             return_predictions=False, return_confusion_matrix=False, eval_label_indices=None):
    """Since Phase 1 review: every evaluation now returns per-class precision,
    recall, AND F1 (not recall alone), per-class predicted-count, and
    optionally the full confusion matrix and raw predictions.

    eval_label_indices: the FIXED, EXPLICIT set of class indices macro_f1 is
    averaged over -- never left to sklearn's default dynamic inference from
    the union of y_true/y_pred. That default is a real bug source: if a class
    absent from y_true (e.g. Password Cracking is absent from the ISOT
    validation split entirely) is nonetheless PREDICTED at least once, an
    unspecified `labels=` argument silently pulls it into the average with
    F1=0, changing the denominator run-to-run depending on what the model
    happened to predict. Defaults to ALL classes in class_names (0..N-1) if
    not given, but ALWAYS passed explicitly to sklearn -- never omitted."""
    model.eval()
    preds = []
    loader = make_loader(X, y, batch_size, shuffle=False)
    total_loss, n = 0.0, 0
    w = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=w, reduction="sum")
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total_loss += criterion(out, yb).item()
        n += len(yb)
        preds.append(out.argmax(dim=1).cpu())
    preds = torch.cat(preds).numpy()
    y_np = y.numpy()
    all_labels = list(range(len(class_names)))
    macro_labels = eval_label_indices if eval_label_indices is not None else all_labels

    macro_f1 = f1_score(y_np, preds, labels=macro_labels, average="macro", zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(y_np, preds, labels=all_labels,
                                                                 average=None, zero_division=0)
    pred_counts = np.bincount(preds, minlength=len(class_names))

    result = {
        "loss": total_loss / n,
        "macro_f1": float(macro_f1),
        "macro_f1_label_indices": list(macro_labels),
        "per_class_recall": {class_names[i]: float(recall[i]) for i in all_labels},
        "per_class_precision": {class_names[i]: float(precision[i]) for i in all_labels},
        "per_class_f1": {class_names[i]: float(f1[i]) for i in all_labels},
        "per_class_predicted_count": {class_names[i]: int(pred_counts[i]) for i in all_labels},
    }
    if return_confusion_matrix:
        cm = confusion_matrix(y_np, preds, labels=all_labels)
        result["confusion_matrix"] = cm.tolist()  # rows=true, cols=predicted, order=class_names
    if return_predictions:
        result["y_true"] = y_np
        result["y_pred"] = preds
    return result

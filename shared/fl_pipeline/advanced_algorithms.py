"""
SCAFFOLD, FedNova, FedAdam -- for Paper 1, Item 2 (DESIGN_FROZEN.md):
"is rare-class forgetting specific to FedAvg, or does it persist under
FL algorithms designed to address heterogeneity and server-side
optimization?" FedYogi and Krum/Trimmed-Mean deliberately excluded per
that design (Byzantine defenses belong to papers 3/4).
"""
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, batch_size, shuffle=True):
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# SCAFFOLD
# ---------------------------------------------------------------------------
def zeros_like_state(model):
    """Always CPU -- c_global/c_local persist across rounds and must match the
    CPU-resident state dicts (global_state_start, new_state) they're combined
    with outside the training loop. Moved to DEVICE only transiently inside
    local_train_scaffold's gradient-correction step."""
    return {k: torch.zeros_like(v, dtype=torch.float32, device="cpu") for k, v in model.state_dict().items()}


def local_train_scaffold(model, X, y, device, c_global, c_local_i, global_state_start, epochs=1,
                          batch_size=4096, lr=1e-3, class_weights=None):
    """Option-II SCAFFOLD local update: gradients corrected by (c_global - c_local_i)
    at every step, then c_local_i is refreshed from the actual drift observed."""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    w = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=w)
    loader = make_loader(X, y, batch_size)
    param_names = [n for n, _ in model.named_parameters()]
    c_g = [c_global[n].to(device) for n in param_names]
    c_i = [c_local_i[n].to(device) for n in param_names]

    n_steps = 0
    last_loss = None
    for _ in range(epochs):
        total_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            with torch.no_grad():
                for p, cgv, civ in zip(model.parameters(), c_g, c_i):
                    p.grad.add_(cgv - civ)
            opt.step()
            n_steps += 1
            total_loss += loss.item(); n_batches += 1
        last_loss = total_loss / max(n_batches, 1)

    new_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    new_c_local_i = {}
    delta_c_i = {}
    for n in param_names:
        upd = c_local_i[n] - c_global[n] + (global_state_start[n] - new_state[n]) / (n_steps * lr)
        new_c_local_i[n] = upd
        delta_c_i[n] = upd - c_local_i[n]
    # non-trainable buffers (none expected in this MLP, but keep state complete)
    for k in new_state:
        if k not in param_names:
            new_c_local_i[k] = c_local_i.get(k, torch.zeros_like(new_state[k]))
            delta_c_i[k] = torch.zeros_like(new_state[k])

    return new_state, delta_c_i, new_c_local_i, n_steps, len(X), last_loss


def aggregate_scaffold(client_states, client_delta_c, weights, c_global):
    agg_state = copy.deepcopy(client_states[0])
    for key in agg_state:
        agg_state[key] = sum(w * cs[key].float() for w, cs in zip(weights, client_states))
    new_c_global = {key: c_global[key] + sum(w * dc[key] for w, dc in zip(weights, client_delta_c))
                     for key in c_global}
    return agg_state, new_c_global


# ---------------------------------------------------------------------------
# FedNova
# ---------------------------------------------------------------------------
def make_optimizer(name, params, lr):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"unknown optimizer {name}")


def local_train_generic(model, X, y, device, epochs=1, batch_size=4096, lr=1e-3,
                         optimizer="adam", class_weights=None):
    """Optimizer-selectable local step, for the Item-2 optimizer-matched
    screening (FedAvg-SGD control, FedAdam's client-side step per the
    original Reddi et al. 2020 protocol, which uses client SGD, not Adam)."""
    model.train()
    opt = make_optimizer(optimizer, model.parameters(), lr)
    w = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=w)
    loader = make_loader(X, y, batch_size)
    n_steps = 0
    last_loss = None
    for _ in range(epochs):
        total_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            n_steps += 1
            total_loss += loss.item(); n_batches += 1
        last_loss = total_loss / max(n_batches, 1)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return state, n_steps, len(X), last_loss


def local_train_fednova(model, X, y, device, epochs=1, batch_size=4096, lr=1e-3,
                         optimizer="sgd", class_weights=None):
    """Also returns tau_i (mini-batch step count), needed to normalize
    heterogeneous local step counts before aggregation. Defaults to SGD --
    the simplest defensible FedNova baseline per the original paper's primary
    experimental setup (Wang et al. 2020)."""
    model.train()
    opt = make_optimizer(optimizer, model.parameters(), lr)
    w = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=w)
    loader = make_loader(X, y, batch_size)
    n_steps = 0
    last_loss = None
    for _ in range(epochs):
        total_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            n_steps += 1
            total_loss += loss.item(); n_batches += 1
        last_loss = total_loss / max(n_batches, 1)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return state, n_steps, len(X), last_loss


def aggregate_fednova(client_states, global_state, weights, taus):
    tau_eff = sum(w * t for w, t in zip(weights, taus))
    new_global = {}
    for key in global_state:
        d = sum(w * (cs[key].float() - global_state[key].float()) / t
                for w, cs, t in zip(weights, client_states, taus))
        new_global[key] = global_state[key].float() + tau_eff * d
    return new_global


# ---------------------------------------------------------------------------
# FedAdam (server-side adaptive optimization, FedOpt family)
# ---------------------------------------------------------------------------
def aggregate_fedadam(client_states, global_state, weights, m_state, v_state,
                       server_lr=0.01, beta1=0.9, beta2=0.99, tau=1e-3):
    delta = {key: sum(w * (cs[key].float().cpu() - global_state[key].float())
                       for w, cs in zip(weights, client_states)) for key in global_state}
    new_m, new_v, new_global = {}, {}, {}
    for key in global_state:
        new_m[key] = beta1 * m_state[key] + (1 - beta1) * delta[key]
        new_v[key] = beta2 * v_state[key] + (1 - beta2) * delta[key].pow(2)
        new_global[key] = global_state[key].float() + server_lr * new_m[key] / (new_v[key].sqrt() + tau)
    return new_global, new_m, new_v

"""Move a trained encoder into a fresh PLANET, and freeze it if asked.

The harness cuts checkpoints on the host (`gnnb encoder split`); this is the other half, and it
has to live in the model's own container because loading weights back needs the model class.

**Where PLANET's boundary is.** The registry declared `head = ["FC"]` until 2026-09-13, a module
this model does not have — the checkpoint carries `proteinegnn`, `ligandgat` and `prolig` and
nothing else, so the split refused and nobody had run it.

The corrected boundary follows the rule that settled SIGN's: the encoder is exactly what
produces the embedding. `embed_complexes.py` replays the forward as far as

    fresidues = model.proteinegnn(...)
    fatoms    = model.ligandgat(...)
    fatoms, fresidues = model.prolig.prolig_attention(...)

and pools those two, so the cross-attention is encoder. What follows it are three prediction
heads — affinity, the protein-ligand contact map, and the intra-ligand distance matrix. All
three are PLANET's simultaneous training targets rather than representations, and carrying them
into a fine-tune would transfer an opinion about contact frequencies along with the encoder.

29 tensors on the encoder side, 20 on the head side.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: The three prediction heads, by parameter-name prefix. Must agree with `encoder.head` for
#: planet.modern in harness/registry.toml — the harness cuts the file and this loads it.
HEAD_PREFIXES = (
    "prolig.linear_lig_interaction",
    "prolig.linear_lig_interaction_last",
    "prolig.linear_pocket_interaction",
    "prolig.linear_ligand_interaction",
    "prolig.pro_lig_interaction",
    "prolig.linear_pocket_affinity",
    "prolig.linear_ligand_affinity",
    "prolig.linear_affinity",
)


def is_head(name: str) -> bool:
    """Whether a parameter belongs to a prediction head.

    Matched at module boundaries, which is load-bearing here rather than theoretical:
    `prolig.linear_lig_interaction` and `prolig.linear_lig_interaction_last` are different
    modules and one is a prefix of the other, so a bare `startswith` over the shorter name
    would swallow the longer. Both are heads, so the result happens to be the same — but
    `prolig.prolig_attention` sitting beside them is *not*, and that is what the rule protects.
    """
    return any(name == p or name.startswith(p + ".") for p in HEAD_PREFIXES)


def read_encoder(path: str) -> dict[str, torch.Tensor]:
    """Load an encoder file, accepting either of the two shapes it arrives in.

    `weights_only=True` is not negotiable: an encoder file is tensors, and opening a foreign
    checkpoint any other way is unprotected unpickling.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("encoder"), dict):
        payload = payload["encoder"]
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a state dict, got {type(payload).__name__}")
    return {k: v for k, v in payload.items() if torch.is_tensor(v)}


def transfer_encoder(model: nn.Module, path: str, freeze: bool = False) -> dict[str, object]:
    """Overlay an encoder onto a fresh model, leaving the heads at their initialisation.

    Loaded **strictly**: a partial load that silently moved nothing is indistinguishable from
    success until the curve disappoints.
    """
    encoder = read_encoder(path)
    target = model.state_dict()

    head_keys = [k for k in encoder if is_head(k)]
    if head_keys:
        raise SystemExit(
            f"{path} carries {len(head_keys)} head tensors ({head_keys[:3]}). That is a whole "
            f"checkpoint, not an encoder — cut it with `gnnb encoder split` first."
        )
    unknown = [k for k in encoder if k not in target]
    if unknown:
        raise SystemExit(f"{path}: {len(unknown)} tensors have no slot in this model: {unknown[:5]}")
    mismatched = [k for k, v in encoder.items() if target[k].shape != v.shape]
    if mismatched:
        raise SystemExit(
            f"{path}: shape mismatch on {len(mismatched)} tensors: "
            + ", ".join(f"{k} {tuple(encoder[k].shape)} into {tuple(target[k].shape)}"
                        for k in mismatched[:3])
            + ". The encoder was trained at a different feature_dims; match the sizes."
        )
    uncovered = [k for k in target if k not in encoder and not is_head(k)]
    if uncovered:
        raise SystemExit(
            f"{path}: {len(uncovered)} model tensors are neither in the encoder nor in the "
            f"head — the boundary is wrong: {uncovered[:5]}"
        )

    model.load_state_dict({**target, **encoder}, strict=True)
    moved = sum(v.numel() for v in encoder.values())
    print(f"transferred {len(encoder)} tensors / {moved:,} params from {path}")

    frozen = 0
    if freeze:
        for name, param in model.named_parameters():
            if not is_head(name):
                param.requires_grad_(False)
                frozen += param.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"froze {frozen:,} encoder params; {trainable:,} trainable in the heads")

    return {"init_encoder": path, "transferred_tensors": len(encoder),
            "transferred_params": moved, "frozen_params": frozen}


def set_training_mode(model: nn.Module, training: bool, frozen_encoder: bool) -> None:
    """Put the model in the right mode, keeping a frozen encoder deterministic.

    PLANET's encoder carries no dropout and no batch norm, so today this is equivalent to
    `model.train(training)`. It is here anyway, in the same shape as SIGN's and SS-GNN's,
    because the rule belongs to the contract rather than to whichever layers a model happens
    to have: a frozen encoder must produce the same representation every step, or
    `--freeze-encoder` is measuring something other than what `gnnb probe` measures.
    """
    if not training or not frozen_encoder:
        model.train(training)
        return
    model.eval()
    for name, module in model.named_modules():
        if name and is_head(name) and any(p.requires_grad for p in module.parameters()):
            module.train()

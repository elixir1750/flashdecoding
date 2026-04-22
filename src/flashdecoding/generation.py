"""Single-example vanilla decoding with basic latency and memory metrics."""

from __future__ import annotations

import time
from typing import Any

import torch

from .metrics import get_peak_memory_bytes, reset_peak_memory, synchronize_if_needed


def _set_seed(seed: int | None) -> None:
    """Seed PyTorch for reproducible decoding when requested."""

    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_next_token(logits: torch.Tensor) -> torch.Tensor:
    """Select the next token id with greedy decoding."""

    return torch.argmax(logits, dim=-1, keepdim=True)


@torch.inference_mode()
def generate_once(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    seed: int | None,
) -> dict[str, Any]:
    """Generate a single continuation with vanilla greedy decoding."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1.")

    _set_seed(seed)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    if input_ids.size(0) != 1:
        raise ValueError("This minimal scaffold only supports batch size 1.")

    reset_peak_memory(device)
    synchronize_if_needed(device)
    total_start = time.perf_counter()

    first_outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    first_logits = first_outputs.logits[0, -1, :]
    next_token = select_next_token(logits=first_logits)
    synchronize_if_needed(device)
    ttft = time.perf_counter() - total_start

    generated_ids = [int(next_token.item())]
    past_key_values = first_outputs.past_key_values
    current_input_ids = next_token.view(1, 1).to(device)
    current_attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
        dim=-1,
    )

    eos_token_id = tokenizer.eos_token_id

    while len(generated_ids) < max_new_tokens:
        if eos_token_id is not None and generated_ids[-1] == eos_token_id:
            break

        step_outputs = model(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = step_outputs.past_key_values
        step_logits = step_outputs.logits[0, -1, :]
        next_token = select_next_token(logits=step_logits)

        generated_ids.append(int(next_token.item()))
        current_input_ids = next_token.view(1, 1).to(device)
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=device)],
            dim=-1,
        )

    synchronize_if_needed(device)
    total_latency = time.perf_counter() - total_start

    peak_memory_bytes, memory_source = get_peak_memory_bytes(device)
    output_ids = torch.tensor(generated_ids, dtype=input_ids.dtype, device=device).view(1, -1)
    full_sequence = torch.cat([input_ids, output_ids], dim=-1)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    full_text = tokenizer.decode(full_sequence[0], skip_special_tokens=True)

    generated_tokens = len(generated_ids)
    tpot = None
    if generated_tokens > 1:
        tpot = (total_latency - ttft) / (generated_tokens - 1)

    return {
        "prompt": prompt,
        "prompt_tokens": int(input_ids.shape[-1]),
        "generated_tokens": generated_tokens,
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        "full_text": full_text,
        "ttft_seconds": ttft,
        "tpot_seconds": tpot,
        "total_latency_seconds": total_latency,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_source": memory_source,
    }

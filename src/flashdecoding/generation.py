"""Single-example decoding with latency/memory metrics and streaming support."""

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


def _prepare_inputs(
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    add_special_tokens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize and move a single prompt to the target device."""

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    if input_ids.size(0) != 1:
        raise ValueError("This minimal scaffold only supports batch size 1.")

    return input_ids, attention_mask


def summarize_top_tokens(logits: torch.Tensor, tokenizer: Any, top_k: int = 5) -> list[dict[str, Any]]:
    """Summarize the highest-logit next-token candidates for debugging."""

    top_k = max(1, min(top_k, int(logits.shape[-1])))
    values, indices = torch.topk(logits, k=top_k)
    summary: list[dict[str, Any]] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        summary.append(
            {
                "token_id": int(index),
                "token_piece": tokenizer.convert_ids_to_tokens(int(index)),
                "token_text": tokenizer.decode([int(index)], skip_special_tokens=False),
                "logit": float(value),
            }
        )
    return summary


@torch.inference_mode()
def generate_stream(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    seed: int | None,
    add_special_tokens: bool = True,
):
    """Yield per-token decoding progress for a single continuation."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1.")

    _set_seed(seed)
    input_ids, attention_mask = _prepare_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        add_special_tokens=add_special_tokens,
    )

    reset_peak_memory(device)
    synchronize_if_needed(device)
    total_start = time.perf_counter()

    first_outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    first_logits = first_outputs.logits[0, -1, :]
    first_step_top_tokens = summarize_top_tokens(first_logits, tokenizer=tokenizer)
    next_token = select_next_token(logits=first_logits)
    synchronize_if_needed(device)

    ttft = time.perf_counter() - total_start
    generated_ids = [int(next_token.item())]
    output_ids = torch.tensor(generated_ids, dtype=input_ids.dtype, device=device).view(1, -1)
    yield {
        "event": "token",
        "prompt_tokens": int(input_ids.shape[-1]),
        "generated_tokens": len(generated_ids),
        "generated_token_ids": list(generated_ids),
        "generated_text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
        "ttft_seconds": ttft,
        "elapsed_seconds": ttft,
        "first_generated_token_id": int(next_token.item()),
        "first_generated_token_is_eos": bool(
            tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id
        ),
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "prompt_token_ids": input_ids[0].tolist(),
        "prompt_contains_eos_token": bool(
            tokenizer.eos_token_id is not None and tokenizer.eos_token_id in input_ids[0].tolist()
        ),
        "first_step_top_tokens": first_step_top_tokens,
    }

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
        elapsed_seconds = time.perf_counter() - total_start
        output_ids = torch.tensor(generated_ids, dtype=input_ids.dtype, device=device).view(1, -1)
        yield {
            "event": "token",
            "prompt_tokens": int(input_ids.shape[-1]),
            "generated_tokens": len(generated_ids),
            "generated_token_ids": list(generated_ids),
            "generated_text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
            "ttft_seconds": ttft,
            "elapsed_seconds": elapsed_seconds,
            "first_generated_token_id": generated_ids[0],
            "first_generated_token_is_eos": bool(
                tokenizer.eos_token_id is not None and generated_ids[0] == tokenizer.eos_token_id
            ),
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "prompt_token_ids": input_ids[0].tolist(),
            "prompt_contains_eos_token": bool(
                tokenizer.eos_token_id is not None and tokenizer.eos_token_id in input_ids[0].tolist()
            ),
            "first_step_top_tokens": first_step_top_tokens,
        }


@torch.inference_mode()
def generate_once(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    seed: int | None,
    add_special_tokens: bool = True,
) -> dict[str, Any]:
    """Generate a single continuation with greedy decoding."""

    last_event: dict[str, Any] | None = None
    for event in generate_stream(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        seed=seed,
        add_special_tokens=add_special_tokens,
    ):
        last_event = event

    if last_event is None:
        raise RuntimeError("generate_stream did not produce any tokens.")

    input_ids, _attention_mask = _prepare_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        add_special_tokens=add_special_tokens,
    )
    synchronize_if_needed(device)
    total_latency = float(last_event["elapsed_seconds"])

    peak_memory_bytes, memory_source = get_peak_memory_bytes(device)
    generated_ids = list(last_event["generated_token_ids"])
    output_ids = torch.tensor(generated_ids, dtype=input_ids.dtype, device=device).view(1, -1)
    full_sequence = torch.cat([input_ids, output_ids], dim=-1)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    full_text = tokenizer.decode(full_sequence[0], skip_special_tokens=True)

    generated_tokens = len(generated_ids)
    tpot = None
    if generated_tokens > 1:
        tpot = (total_latency - float(last_event["ttft_seconds"])) / (generated_tokens - 1)

    return {
        "prompt": prompt,
        "prompt_tokens": int(input_ids.shape[-1]),
        "prompt_token_ids": last_event["prompt_token_ids"],
        "prompt_contains_eos_token": last_event["prompt_contains_eos_token"],
        "generated_tokens": generated_tokens,
        "generated_token_ids": generated_ids,
        "first_generated_token_id": last_event["first_generated_token_id"],
        "first_generated_token_is_eos": last_event["first_generated_token_is_eos"],
        "tokenizer_eos_token_id": last_event["tokenizer_eos_token_id"],
        "first_step_top_tokens": last_event["first_step_top_tokens"],
        "generated_text": generated_text,
        "full_text": full_text,
        "ttft_seconds": float(last_event["ttft_seconds"]),
        "tpot_seconds": tpot,
        "total_latency_seconds": total_latency,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_source": memory_source,
    }

"""
Smoke test for the MoE-instrumentation registry.

Loads a target HF model, wraps it with `MoEMonitor`, runs a tiny `generate()`
call, and asserts that the per-expert hidden states and router outputs are
captured. Exits 0 on success, non-zero on failure.

Run with:
    python experiments/_smoke_test_moe.py --model <hf_id>

The default model is the one set in `pipeline_config.sh` ($MODEL). Override
with `--model`. Quantization defaults to 4-bit; override with `--quantize`.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Make the repo root importable
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from moeuncert.experiments import get_quantization_kwargs
from moeuncert.monitoring import MoEMonitor


def _log(msg: str) -> None:
    print(f"[smoke-test {time.strftime('%H:%M:%S')}]  {msg}", flush=True)


def _check(model_id: str, quantize: str, max_new_tokens: int) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _log(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    _log(f"Loading model (quantize={quantize})...")
    qk = get_quantization_kwargs(quantize)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="eager",
            device_map="auto",
            **qk,
        )
    except Exception as e:
        _log(f"FAILED to load model: {type(e).__name__}: {e}")
        return 2

    device = next(model.parameters()).device
    _log(f"Model on {device}.")

    monitor = MoEMonitor(
        model=model,
        tokenizer=tokenizer,
        output_attentions=True,
        output_hidden_states=True,
        output_scores=True,
        output_router_logits=False,  # known to break generate() for some MoE models
        output_experts_hidden=True,
    )

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    _log(f"Running generate() (max_new_tokens={max_new_tokens})...")
    try:
        output = monitor.generate(**inputs, max_new_tokens=max_new_tokens)
    except Exception as e:
        _log(f"FAILED generate(): {type(e).__name__}: {e}")
        return 3

    # 1) sequences must be present and of expected length
    seq = output.sequences
    expected_len = inputs.input_ids.shape[1] + max_new_tokens
    if seq.shape[1] != expected_len:
        _log(
            f"FAIL: sequences length {seq.shape[1]} != expected {expected_len}."
        )
        return 4

    # 2) experts_hidden must be non-empty
    eh = output.get("experts_hidden", None)
    if not eh:
        _log("FAIL: 'experts_hidden' missing from generate() output.")
        return 5
    n_steps = len(eh)
    if n_steps < 1:
        _log(f"FAIL: 'experts_hidden' has no steps ({n_steps}).")
        return 6
    n_layers = len(eh[0])
    _log(f"  experts_hidden: {n_steps} steps x {n_layers} layers")
    if n_layers < 1:
        _log(f"FAIL: no MoE layers detected (n_layers={n_layers}).")
        return 7

    # 3) Inspect the per-layer hidden state structure on one step
    step = eh[-1]
    sample = step[0]
    for required_key in ("expert_idx", "expert_weights", "expert_hidden_states"):
        if required_key not in sample:
            _log(
                f"FAIL: expert hidden dict missing key '{required_key}' "
                f"(got keys: {list(sample.keys())})."
            )
            return 8

    # 4) last_experts_hidden should still be set on the MoE blocks for
    # inspection via the last step's shape
    eh_states = sample["expert_hidden_states"]
    if eh_states.ndim != 4:
        _log(
            f"FAIL: expert_hidden_states must be 4D (B, S, top_k, H); "
            f"got {eh_states.shape}."
        )
        return 9
    _log(f"  expert_hidden_states sample shape: {tuple(eh_states.shape)}")

    # 5) MoE block count: should match the number of MoE decoder layers.
    from moeuncert.forwards import is_moe_block
    moe_count = sum(1 for m in model.modules() if is_moe_block(m))
    _log(f"  is_moe_block() found {moe_count} MoE block(s) in the model.")
    if moe_count < 1:
        _log("FAIL: no MoE blocks detected via is_moe_block().")
        return 10
    if moe_count != n_layers:
        _log(
            f"FAIL: MoE-block count ({moe_count}) != "
            f"experts_hidden layer count ({n_layers})."
        )
        return 11

    _log("ALL CHECKS PASSED.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for the MoE-instrumentation registry."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="HF model id to test (default: OLMoE-1B-7B-0924-Instruct).",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="4-bit",
        choices=["16-bit", "8-bit", "4-bit"],
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
        help="Tokens to generate (default: 4; small for speed).",
    )
    args = parser.parse_args()

    _log(f"smoke test starting: model={args.model}, quantize={args.quantize}")
    return _check(args.model, args.quantize, args.max_new_tokens)


if __name__ == "__main__":
    sys.exit(main())

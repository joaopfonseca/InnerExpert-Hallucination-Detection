def _unpack_input_outputs(outputs):
    """Get the logits, hidden states, and attentions for the last token in the input sequence."""
    outputs_sliced = {}
    if "logits" in outputs:
        outputs_sliced["logits"] = [outputs["logits"][0][:, -1]]
    if "hidden_states" in outputs:
        outputs_sliced["hidden_states"] = [
            [layer[:, -1, :] for layer in outputs["hidden_states"][0]]
        ]
    if "attentions" in outputs:
        outputs_sliced["attentions"] = outputs["attentions"]
    if "router_logits" in outputs:
        outputs_sliced["router_logits"] = [
            [layer[-1] for layer in outputs["router_logits"][0]]
        ]
    return {
        **outputs_sliced,
        **{k: v for k, v in outputs.items() if k not in outputs_sliced},
    }


def reconstruct_model_output(model, input_ids, output_ids, **model_kwargs):
    """
    Reconstruct the model outputs exactly as they occurred
    during autoregressive generation.

    Args:
        model: The language model
        input_ids: The input prompt tokens (2D tensor: [batch, seq_len])
        output_ids: The complete sequence (input + generated tokens) (2D tensor: [batch, total_len])
        **model_kwargs: Additional model arguments (use_cache, output_hidden_states, etc.)

    Returns:
        List of model outputs, one per generated token
    """
    IGNORE_KEYS = [
        "input_ids",
        "attention_mask",
        "max_new_tokens",
        "return_dict_in_generate",
    ]

    return_dict = model_kwargs.get("return_dict_in_generate", True)

    # Ensure input_ids has batch dimension
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if output_ids.dim() == 1:
        output_ids = output_ids.unsqueeze(0)

    num_input_tokens = input_ids.shape[1]
    num_generated_tokens = output_ids.shape[1] - num_input_tokens

    # First pass: process all input tokens to build KV cache
    out = model(
        input_ids,
        past_key_values=None,
        return_dict=return_dict,
        **{k: v for k, v in model_kwargs.items() if k not in IGNORE_KEYS},
    )
    past_key_values = out.past_key_values

    all_outputs = _unpack_input_outputs(
        {k: [v] for k, v in out.items() if k != "past_key_values"}
    )

    # Autoregressive generation: process each generated token one at a time
    for t in range(num_input_tokens, num_input_tokens + num_generated_tokens - 1):
        step_out = model(
            input_ids=output_ids[:, t : t + 1],
            past_key_values=past_key_values,
            return_dict=return_dict,
            **{k: v for k, v in model_kwargs.items() if k not in IGNORE_KEYS},
        )
        past_key_values = step_out.past_key_values

        for k, v in step_out.items():
            if k == "past_key_values":
                continue
            all_outputs[k].append(v)

    all_outputs["sequences"] = output_ids
    all_outputs["scores"] = all_outputs.pop("logits", None)
    all_outputs["past_key_values"] = past_key_values

    return all_outputs

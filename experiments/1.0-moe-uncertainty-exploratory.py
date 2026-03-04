"""
This script serves as an exploratory playground for testing the collection of
MoE signals and the reconstruction of model outputs.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from moeuncert import llm_description, generate_params, reconstruct_model_output

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMoE-1B-7B-0924-Instruct", attn_implementation="eager"
).to(DEVICE)

print(llm_description(model))

messages = [
    {
        "role": "user",
        # "content": "How do Mixture-of-Experts LLM models handle uncertainty?",
        "content": "I want to wash my car and the car wash is only 200 feet away. Should I start my car and drive there or just walk.",
    },
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# NOTE: output_router_logits must be False for `generate`; This is a known limitation.
# See: https://github.com/huggingface/transformers/issues/30731
gen_params = generate_params(
    inputs, tokenizer, max_new_tokens=60, output_router_logits=False
)
outputs = model.generate(**gen_params)

print(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0])

# This is the same as outputs, but with the hidden states reconstructed exactly as they occurred
# during autoregressive generation.
gen_params_rec = generate_params(inputs, tokenizer, max_new_tokens=60)
outputs_rec = reconstruct_model_output(
    model,
    input_ids=inputs["input_ids"],
    output_ids=outputs.sequences,
    **{
        k: v
        for k, v in gen_params_rec.items()
        if k not in ["input_ids", "max_new_tokens"]
    },
)

# Verify that the reconstructed sequence matches the original generated sequence
k = 3
topk_ids = torch.stack(
    [torch.topk(s.squeeze(), k=k).indices for s in outputs["scores"]]
)
topk_ids_rec = torch.stack(
    [torch.topk(s.squeeze(), k=k).indices for s in outputs_rec["scores"]]
)

print(tokenizer.batch_decode(topk_ids[:, 0], skip_special_tokens=False)[0])
print(tokenizer.batch_decode(topk_ids_rec[:, 0], skip_special_tokens=False)[0])
print(
    int((topk_ids == topk_ids_rec).sum()),
    "out of",
    topk_ids.numel(),
    "top k tokens match between original and reconstructed outputs.",
)

#######################################################################################
# Testing the intermediate expert hidden states

print("Intermediate expert hidden states:")

from moeuncert._experts_states import modify_model, reset_model  # , modify_moe_block

modify_model(model)

inputs = tokenizer(
    "I want to wash my car and the car wash is only 200 feet away. Should I start my car and drive there or just walk.",
    return_tensors="pt",
).to(model.device)
_ = model(**inputs)

experts_hidden = model.model.layers[
    7
].mlp.last_experts_hidden  # {num_experts, batch, seq, hidden]

reset_model(model)

#######################################################################################
# Testing with the reconstructed outputs to see if we can also reconstruct the
# intermediate expert hidden states

modify_model(model)

gen_params_rec2 = generate_params(inputs, tokenizer, max_new_tokens=60)
outputs_rec2 = reconstruct_model_output(
    model,
    input_ids=inputs["input_ids"],
    output_ids=outputs.sequences,
    **{
        k: v
        for k, v in gen_params_rec.items()
        if k not in ["input_ids", "max_new_tokens"]
    },
)

reset_model(model)

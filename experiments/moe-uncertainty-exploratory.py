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


#######################################################################################
# Test using hooks to capture the intermediate expert hidden states

experts_activations = {}


def make_hook(layer_idx):
    def hook(module, input, output):
        experts_activations[layer_idx] = module.last_experts_hidden

    return hook


for i, layer in enumerate(model.model.layers):
    layer.mlp.register_forward_hook(make_hook(i))


#######################################################################################
# Modifying the forward method to output the intermediate expert hidden states
# (maybe without using hooks?), using an output_experts_hidden flag to control
# whether to save the expert hidden states or not.
from moeuncert._experts_states import forward_olmoe

# Test with output_experts_hidden=False (default behavior, no overhead)
print("\n--- Test with output_experts_hidden=False ---")
inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
outputs_no_experts = model(**inputs)
print(f"Output shape: {outputs_no_experts.logits.shape}")
print(f"Has experts_hidden: {hasattr(outputs_no_experts, 'experts_hidden')}")

# Test with output_experts_hidden=True (capture expert hidden states)
print("\n--- Test with output_experts_hidden=True ---")
# Note: We need to call the model in a way that propagates the output_experts_hidden flag
# For now, we'll directly access the MoE block to demonstrate
moe_block = model.model.layers[0].mlp
test_hidden = torch.randn(1, 5, model.config.hidden_size, dtype=model.dtype).to(
    model.device
)
result_with_experts = moe_block.forward(test_hidden, output_experts_hidden=True)
if hasattr(moe_block, "last_experts_hidden"):
    print(
        f"Captured experts hidden states for {len(moe_block.last_experts_hidden)} experts"
    )
    for expert_idx, hidden in moe_block.last_experts_hidden.items():
        print(f"  Expert {expert_idx}: shape {hidden.shape}")
else:
    print("No expert hidden states captured")

print("\n--- Full model generation with expert tracking ---")
# For full model generation, we need to explicitly set output_experts_hidden=True
# Since we can't easily pass this through the full model forward, we'll temporarily
# modify the lambda to always use True, run inference, then reset

# Save current forwards
original_forwards = {}
for i, layer in enumerate(model.model.layers):
    original_forwards[i] = layer.mlp.forward

# Set all MoE blocks to capture expert states
for i, layer in enumerate(model.model.layers):
    layer.mlp.forward = lambda hidden_states, module=layer.mlp: forward_olmoe(
        module, hidden_states, output_experts_hidden=True
    )

# Run inference
inputs = tokenizer("Mixture of experts models are", return_tensors="pt").to(
    model.device
)
_ = model(**inputs)

# Check captured expert states
print("Expert activations captured across layers:")
for i, layer in enumerate(model.model.layers):
    if hasattr(layer.mlp, "last_experts_hidden") and layer.mlp.last_experts_hidden:
        print(f"  Layer {i}: {len(layer.mlp.last_experts_hidden)} experts activated")

# Restore original forwards
for i, layer in enumerate(model.model.layers):
    layer.mlp.forward = original_forwards[i]

print("\n✓ Script completed successfully!")
print("Summary: The output_experts_hidden flag allows efficient control over")
print("whether to capture intermediate expert hidden states, avoiding overhead")
print("when not needed while enabling detailed analysis when required.")

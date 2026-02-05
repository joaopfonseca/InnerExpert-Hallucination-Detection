from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from utils import llm_description, generate_params, reconstruct_model_output

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMoE-1B-7B-0924-Instruct", attn_implementation="eager"
).to(DEVICE)

print(llm_description(model))

messages = [
    {"role": "user", "content": "How do Mixture-of-Experts LLM models handle uncertainty?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

gen_params = generate_params(inputs, tokenizer, max_new_tokens=60)
outputs = model.generate(**gen_params, top_k=1)

print(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0])

# This is the same as outputs, but with the hidden states reconstructed exactly as they occurred 
# during autoregressive generation.
outputs_rec = reconstruct_model_output(
    model, 
    input_ids=inputs["input_ids"], 
    output_ids=outputs.sequences,
    **{k: v for k, v in gen_params.items() if k not in ["input_ids", "max_new_tokens"]}
)

# Verify that the reconstructed sequence matches the original generated sequence
k = 3
topk_ids = torch.stack([torch.topk(s.squeeze(), k=k).indices for s in outputs["scores"]])
topk_ids_rec = torch.stack([torch.topk(s.squeeze(), k=k).indices for s in outputs_rec["logits"]])

print(tokenizer.batch_decode(topk_ids[:, 0], skip_special_tokens=False)[0])
print(tokenizer.batch_decode(topk_ids_rec[:, 0], skip_special_tokens=False)[0])
print(int((topk_ids == topk_ids_rec).sum()), "out of", topk_ids.numel(), "top k tokens match between original and reconstructed outputs.")

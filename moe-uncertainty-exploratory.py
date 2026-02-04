from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from utils import llm_description

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_params(inputs, tokenizer, **kwargs):
    return {
        **inputs,
        "attention_mask":inputs["attention_mask"],
        # "temperature":.5, 
        # "top_p":0.9,
        # "do_sample":True,
        "pad_token_id":tokenizer.eos_token_id,
        "return_dict_in_generate":True,
        "output_attentions":True,
        "output_hidden_states":True,
        "output_scores":True,
        "use_cache":True,
        **kwargs
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
    all_outputs = {}
    
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
        use_cache=True,
        return_dict=True,
        **{k: v for k, v in model_kwargs.items() if k not in ["input_ids", "attention_mask", "max_new_tokens"]}
    )
    past_key_values = out.past_key_values
    
    # Autoregressive generation: process each generated token one at a time
    for t in range(num_generated_tokens):
        token_idx = num_input_tokens + t
        out = model(
            output_ids[:, token_idx:token_idx+1],  # one token at a time
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            **{k: v for k, v in model_kwargs.items() if k not in ["input_ids", "attention_mask", "max_new_tokens"]}
        )
        past_key_values = out.past_key_values

        for k, v in out.items():
            if k not in all_outputs:
                all_outputs[k] = []
            all_outputs[k].append(v)
    
    return all_outputs


tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct", attn_implementation="eager").to(DEVICE)

llm_description(model)

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

# content = model.generate(
#     **inputs,
#     max_new_tokens=60
# )
# 
# print(tokenizer.decode(content[0][:inputs["input_ids"].shape[-1]], skip_special_tokens=True))
# print("------------------")
# print(tokenizer.decode(content[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
# 
# outputs_rec1 = model(
#     outputs1,
#     **generate_params(inputs, tokenizer, max_new_tokens=60)
# )

outputs = model.generate(
    **generate_params(inputs, tokenizer, max_new_tokens=60),
)

# This is the same as outputs, but with the hidden states reconstructed exactly as they occurred 
# during autoregressive generation.
outputs_rec = reconstruct_model_output(
    model, 
    input_ids=inputs["input_ids"], 
    output_ids=outputs.sequences,
    output_hidden_states=True,
    use_cache=True
)

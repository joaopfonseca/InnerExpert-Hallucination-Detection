import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from moeuncert.monitoring import MoEMonitor
from moeuncert.datasets import fetch_realtimeqa
from moeuncert.utils import tokenize_realtimeqa, standardize_outputs, move_to_device

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {DEVICE}")

df = fetch_realtimeqa(split="latest")
question_sample = df.iloc[rng.choice(len(df))]

model_name = "allenai/OLMoE-1B-7B-0924-Instruct"
model_slug = model_name.replace("/", "__")

# Clear cache to ensure we have enough memory for the model
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Required for batching
tokenizer.padding_side = "left"  # Left padding for generation

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",
    torch_dtype=torch.float16,
    device_map="auto",
)
model_monitor = MoEMonitor(model=model, tokenizer=tokenizer, output_router_logits=False)

questions_tokenized = tokenize_realtimeqa(
    tokenizer, question_sample.to_frame().T, with_evidence="both"
)
questions_tokenized = move_to_device(questions_tokenized, device=DEVICE)

outputs = model_monitor.generate(**questions_tokenized, max_new_tokens=65)
outputs = standardize_outputs(outputs, device="cpu")

print(
    "\nNO EVIDENCE PROVIDED:\n=====================\n",
    tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)[0],
)
print(
    "\nWITH EVIDENCE:\n==============\n",
    tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)[1],
)

###############################################################################
# Computing monitoring metrics

from moeuncert.metrics._metrics import hidden_score, attention_score, topk_entropy

# Shape of hidden states: (batch_size, sequence_length, n_layers, hidden_size)
# Shape of hidden scores: (batch_size, sequence_length, n_layers)
hidden_scores = hidden_score(outputs["hidden_states"])

# Shape of attention matrices: (batch_size, n_layers, num_heads, seq_len, seq_len)
# Shape of attention scores: (batch_size, n_layers, num_heads, seq_len)
attention_scores = attention_score(outputs["attentions"])

# Shape of output logits: (batch_size, sequence_length, vocab_size)
# Shape of top-k entropy scores: (batch_size, sequence_length)
scores_entropy = topk_entropy(outputs["scores"], k=5)

# Shape of expert weights: (batch_size, sequence_length, n_layers, n_experts)
# Shape of router entropy scores: (batch_size, sequence_length, n_layers)
router_entropy = topk_entropy(outputs["expert_weights"], softmax=False)

# Shape of expert hidden states: (batch_size, sequence_length, n_layers, top_k, hidden_size)
outputs["expert_hidden_states"]

outputs["expert_idx"]


###############################################################################
# Visual examples
pad_mask = outputs["sequences"] != tokenizer.pad_token_id  # (batch_size, sequence_length)
question_length = questions_tokenized["input_ids"].shape[-1]

plt.plot(router_entropy[0, question_length:, 12][pad_mask[0, question_length+1:]], label="no evidence")
plt.plot(router_entropy[1, question_length:, 12][pad_mask[1, question_length+1:]], label="evidence-based")
plt.legend()
plt.title("Router entropy for layer 12")
plt.xlabel("Token position")
plt.ylabel("Router entropy")
plt.show()

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from moeuncert.datasets import fetch_realtimeqa
from moeuncert.utils import generate_params
import torch
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
tokenizer.pad_token = tokenizer.eos_token  # Required for batching
tokenizer.padding_side = "left"  # Left padding for generation
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMoE-1B-7B-0924-Instruct", attn_implementation="eager"
).to(DEVICE)

# Load a small sample of the RealtimeQA dataset
df = fetch_realtimeqa(split=2025, month=12).tail(10)


messages = [
    {
        "role": "system", 
        "content": "You are a helpful assistant who provides accurate and very concise answers to questions about recent events. We are currently in February 2026."
    },
    {"role": "user", "content": df.iloc[0]["question_sentence"]}
], 
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
    inputs, 
    tokenizer, 
    max_new_tokens=65, 
    output_attentions=False,
    output_hidden_states=False,
    output_scores=False,
    output_router_logits=False,
)
outputs = model.generate(**gen_params)

print(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0])

###############################################################################
# Batch generation for all questions in the dataset

dataset = Dataset.from_pandas(df)

def tokenize_function(examples):
    """Tokenize questions with chat template."""
    texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant who provides accurate and concise answers to questions about recent events. We are currently in February 2026."},
                {"role": "user", "content": q}
            ],
            add_generation_prompt=True,
            tokenize=False
        )
        for q in examples["question_sentence"]
    ]
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
tokenized.set_format(type="torch")

all_outputs = {}
with torch.no_grad():
    for i in tqdm(list(range(0, len(tokenized), 64))):
        batch = tokenized[i:i+64]
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        gen_params = generate_params(
            batch, 
            tokenizer, 
            max_new_tokens=65, 
            output_attentions=False,
            output_hidden_states=True,
            output_scores=False,
            output_router_logits=False,
        )

        outputs = model.generate(**gen_params, do_sample=False)
        for key, output in outputs.items():
            if key not in all_outputs:
                all_outputs[key] = output.cpu()
            all_outputs[key]

outputs_dataset = Dataset.from_dict({"generated_ids": all_outputs})
outputs_dataset.save_to_disk("realtimeqa_generated_outputs")

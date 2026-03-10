from ._generation_reconstruction import reconstruct_model_output
from ..forwards import modify_model, reset_model
from ..utils import standardize_outputs
from ..metrics import (
    hidden_score,
    attention_score,
    topk_entropy,
    expert_hidden_scores,
    expert_similarity_score,
    expert_usage_frequency,
)
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock


class MoEMonitor:
    def __init__(
        self,
        model,
        tokenizer,
        output_attentions=True,
        output_hidden_states=True,
        output_scores=True,
        output_router_logits=True,
        output_experts_hidden=True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.output_scores = output_scores
        self.output_router_logits = output_router_logits
        self.output_experts_hidden = output_experts_hidden

    def _monitor_kwargs(self):
        return {
            "pad_token_id": self.tokenizer.eos_token_id,
            "output_attentions": self.output_attentions,
            "output_hidden_states": self.output_hidden_states,
            "output_scores": self.output_scores,
            "output_router_logits": self.output_router_logits,
            "return_dict_in_generate": True,
            "use_cache": True,
        }

    def reconstruct_model_output(
        self,
        input_ids,
        output_ids,
        **model_kwargs,
    ):
        if self.output_experts_hidden:
            modify_model(self.model)

        forward_kwargs = self._monitor_kwargs()
        forward_kwargs.update(model_kwargs)

        output = reconstruct_model_output(
            self.model, input_ids, output_ids, **forward_kwargs
        )

        if self.output_experts_hidden:
            reset_model(self.model)

        return output

    def generate(self, **model_kwargs):

        if self.output_experts_hidden:
            modify_model(self.model)

        forward_kwargs = self._monitor_kwargs()
        forward_kwargs.update(model_kwargs)

        if self.output_experts_hidden:
            moe_blocks = [
                m for m in self.model.modules() if isinstance(m, OlmoeSparseMoeBlock)
            ]
            num_layers = len(moe_blocks)
            _step_buffer = []
            _all_steps = []

            def make_moe_hook():
                def hook(module, input, output):
                    _step_buffer.append(module.last_experts_hidden)
                    if len(_step_buffer) == num_layers:
                        _all_steps.append(list(_step_buffer))
                        _step_buffer.clear()

                return hook

            hooks = [m.register_forward_hook(make_moe_hook()) for m in moe_blocks]

        output = self.model.generate(**forward_kwargs)

        if self.output_experts_hidden:
            for h in hooks:
                h.remove()
            output["experts_hidden"] = _all_steps
            reset_model(self.model)

        return output

    def forward(
        self,
        *args,
        **kwargs,
    ):
        if self.output_experts_hidden:
            modify_model(self.model)

        forward_kwargs = self._monitor_kwargs()
        forward_kwargs.update(kwargs)

        output = self.model.forward(*args, **forward_kwargs)

        if self.output_experts_hidden:
            reset_model(self.model)

        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def compute_metrics(self, outputs):
        outputs = standardize_outputs(outputs)

        metrics = {}
        if "hidden_states" in outputs:
            # Shape of hidden states: (batch_size, sequence_length, n_layers, hidden_size)
            # Shape of hidden scores: (batch_size, sequence_length, n_layers)
            metrics["hidden_scores"] = hidden_score(outputs["hidden_states"])

        if "attentions" in outputs:
            # Shape of attention matrices: (batch_size, n_layers, num_heads, seq_len, seq_len)
            # Shape of attention scores: (batch_size, n_layers, num_heads, seq_len)
            metrics["attention_scores"] = attention_score(outputs["attentions"])

        if "scores" in outputs:
            # Shape of output logits: (batch_size, sequence_length, vocab_size)
            # Shape of top-k entropy scores: (batch_size, sequence_length)
            metrics["scores_entropy"] = topk_entropy(outputs["scores"], k=5)

        if "expert_weights" in outputs:
            # Shape of expert weights: (batch_size, sequence_length, n_layers, n_experts)
            # Shape of router entropy scores: (batch_size, sequence_length, n_layers)
            # NOTE: expert weights do not sum to 1 if
            #       model.model.layers[...].mlp.norm_topk_prob is False
            metrics["router_entropy"] = topk_entropy(
                outputs["expert_weights"], softmax=False
            )
        
        if "expert_hidden_states" in outputs and "expert_weights" in outputs:
            # Shape of expert hidden states: 
            # (batch_size, sequence_length, n_layers, n_experts, hidden_size)
            # Option 1: sum hidden state score over experts, weighing by the expert weights
            metrics["expert_hidden_scores"] = expert_hidden_scores(
                outputs["expert_hidden_states"], outputs["expert_weights"]
            )
            
            # Option 2: weighted sum of cosine similarity among expert hidden states
            metrics["expert_similarities"] = expert_similarity_score(
                outputs["expert_hidden_states"], outputs["expert_weights"]
            )
        
        if "expert_idx" in outputs:
            # Check usage frequency of each expert
            expert_idx = outputs["expert_idx"][:, outputs["input_ids"].shape[-1]:]
            metrics["expert_usage"] = expert_usage_frequency(expert_idx)

        return metrics

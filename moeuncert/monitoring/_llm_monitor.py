from ._generation_reconstruction import reconstruct_model_output
from ..forwards import modify_model, reset_model


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

        if self.output_experts_hidden:
            self._monitor_experts_hidden = self._monitor_experts_hidden

    def _monitor_experts_hidden(self):
        modify_model(self.model)
        raise NotImplementedError

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

        output = self.model.generate(**forward_kwargs)

        if self.output_experts_hidden:
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

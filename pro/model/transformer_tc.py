import torch.nn as nn
from transformers.modeling_outputs import ModelOutput

class TextClassification(nn.Module):
    def __init__(self, pretrained_model, num_categories=2, loss_function=None):
        super().__init__()
        self.bert = pretrained_model
        self.hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(self.hidden_size, num_categories)
        self.loss_function = loss_function
    
    def forward(self,
                input_ids,
                attention_mask=None,
                position_ids=None,
                token_type_ids=None,
                output_attentions=False,
                output_hidden_states=False,
                label=None):
        
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            token_type_ids=token_type_ids,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states)
        
        state = outputs.last_hidden_state[:, 0, :]
        state = self.linear(state)
        
        loss = None
        if label is not None and self.loss_function is not None:
            loss = self.loss_function(state, label)
        
        attentions = None
        if output_attentions:
            attentions=outputs.attentions
        
        hidden_states = None
        if output_hidden_states:
            hidden_states=outputs.hidden_states
        
        return ModelOutput(
            logits=state,
            loss=loss,
            last_hidden_state=outputs.last_hidden_state,
            attentions=attentions,
            hidden_states=hidden_states
        )


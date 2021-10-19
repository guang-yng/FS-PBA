"""Auto RoBERTa model"""

import torch
import torch.nn as nn
from packaging import version

from transformers import RobertaForMaskedLM, PreTrainedModel
from .model import PromptRobertaForMaskedLM


class AutoRobertaForMaskedLM(nn.Module):
    def __init__(self, config, use_prompt, model_name_or_path=None):
        super().__init__()

        self.use_prompt = use_prompt
        if model_name_or_path is not None:
            if use_prompt :
                self.roberta = PromptRobertaForMaskedLM.from_pretrained(model_name_or_path, config=config)
            else:
                self.roberta = RobertaForMaskedLM.from_pretrained(model_name_or_path, config=config)
        else:
            if use_prompt:
                self.roberta = PromptRobertaForMaskedLM(config=config)
            else:
                self.roberta = RobertaForMaskedLM(config=config)

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        self.num_labels = config.num_labels

    def forward(self, input_ids=None, attention_mask=None, mask_pos=None, labels=None):
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        output = self.roberta(input_ids, attention_mask=attention_mask)
        prediction_scores = output[1]
        prediction_mask_scores = prediction_scores[torch.arange(prediction_scores.size(0)), mask_pos]

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output



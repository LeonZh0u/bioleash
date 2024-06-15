import os
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

class LMModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        os.environ["http_proxy"] = "http://devproxy.bloomberg.com:82"
        os.environ["https_proxy"] = "http://devproxy.bloomberg.com:82"
        self.config = AutoConfig.from_pretrained(model_name, num_labels=3)
        self.lm = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        # if torch.__version__ >= "2.0" and config.compile:
        #     print("Compiling model...")
        #     opt_model = torch.compile(model)

        # if torch.__version__ >= "2.0" and config.flash_attn:
        #     print("Using Flash Attention")
        #     if config.compile:
        #         opt_model = BetterTransformer.transform(opt_model)
        #     else:
        #         opt_model = BetterTransformer.transform(model)
        # unset proxies
        _ = os.environ.pop('http_proxy', None)
        _ = os.environ.pop('https_proxy', None)
        
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, batch):
        last_hidden_state = self.lm(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).last_hidden_state
        logits = self.classifier(
            self.dropout(last_hidden_state[:, 0])
        )
        return {
            "logits": logits,
        }

    def calculate_loss(self, batch):
        output = self.forward(batch)
        loss = self.loss_fn(output["logits"], batch["labels"].float())
        output["loss"] = loss
        return output

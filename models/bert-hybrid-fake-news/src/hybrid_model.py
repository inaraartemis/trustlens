from transformers import BertModel
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.fc = nn.Linear(768 + feature_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.pooler_output

        combined = torch.cat((cls, features), dim=1)
        return self.sigmoid(self.fc(combined))
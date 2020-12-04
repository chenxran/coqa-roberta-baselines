import torch
from torch import nn
from torch.nn import functional as f
from transformers import RobertaForQuestionAnswering, RobertaConfig, BertForQuestionAnswering, BertConfig


#TODO: 设置参数，使得能够使用roberta/bert等等进行调参

class BertForCoQA(nn.Module):
    def __init__(self, config):
        super(BertForCoQA, self).__init__()
        self.config = config
        self.model = BertForQuestionAnswering.from_pretrained(
            "/nfsshare/home/chenxingran/projects/coqa-roberta-baselines/models/bert/bert-large-uncased-pytorch_model.bin", 
            config=BertConfig.from_pretrained("bert-large-uncased")
        )

    def forward(self, ex):
        input_ids = ex['input_ids']
        attention_mask = ex['attention_mask']
        token_type_ids = ex['token_type_ids']
        start_positions = ex['targets'][:, 0].squeeze().long()
        end_positions = ex['targets'][:, 1].squeeze().long()

        # print(input_ids.size(), attention_mask.size(), start_positions.size(), end_positions.size())

        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=True
            )

        return {'loss': outputs.loss,
                'score_s': outputs.start_logits,
                'score_e': outputs.end_logits,
                'targets': ex['targets']}
        
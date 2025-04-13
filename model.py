import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoTokenizer, AutoModel

config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

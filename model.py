from ast import Add
import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoTokenizer, AutoModel
from tokenizers import AddedToken


#config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
#model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

tokenizer = AutoTokenizer.from_pretrained(
    './kmer_tokenizers/3mer.model',
    padding_side="right",
    use_fast=True,
    trust_remote_code=True,
)

t_t5 = AutoTokenizer.from_pretrained('google-t5/t5-small')

s = "AGC GCT CTN NNT TTN TAG NNN TAG"

def gen_seq(i, k):
    s = ""
    DNA = "TAGCN"
    for p in range(k):
        s += DNA[int(i % 5)]
        i //= 5
    return s

def kmer_iterator(k):
    for i in range(5 ** k):
        yield gen_seq(i, k)
"""
t = tokenizer.train_new_from_iterator([], vocab_size=3)

for w in kmer_iterator(3):
    t.add_tokens(AddedToken(w))
"""

print(tokenizer([s], return_tensors='pt').input_ids.dtype)
print(t_t5([s], return_tensors='pt').input_ids.dtype)

a = torch.tensor([1, 5, 10])
print(torch.ones_like(a).dtype)


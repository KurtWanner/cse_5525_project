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

s = "AGC GCT CTN NNT TTN TAG NNN TAG"
x = tokenizer(s.split(), is_split_into_words=True)
print(x.input_ids)
print(len(tokenizer))

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

t = tokenizer.train_new_from_iterator([], vocab_size=3)

for w in kmer_iterator(3):
    t.add_tokens(AddedToken(w))

print(len(t))

print(s)
print(t([s], is_split_into_words=True).input_ids)

t.save_pretrained('./kmer_tokenizers/3mer.model')


import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoTokenizer, AutoModel

#config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
#model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    padding_side="right",
    use_fast=True,
    trust_remote_code=True,
)

s = "AGC GCT CTN TNN NNT"
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

t = tokenizer.train_new_from_iterator(kmer_iterator(3), vocab_size=5 ** 3 + 1)

print(len(t))

print(s)
print(t(s.split(), is_split_into_words=True).input_ids)


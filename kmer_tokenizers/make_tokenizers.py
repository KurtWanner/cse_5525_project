from ast import Add
import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoTokenizer, AutoModel
from tokenizers import AddedToken


base = AutoTokenizer.from_pretrained('3mer.model')


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



def make_t(k):
    new_t = base.train_new_from_iterator([], vocab_size=3)
    for w in kmer_iterator(k):
        new_t.add_tokens(AddedToken(w))

    save_location = str(k) + 'mer.model'

    new_t.save_pretrained(save_location)

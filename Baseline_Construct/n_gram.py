import pandas as pd
from collections import defaultdict, Counter
import random

df = pd.read_csv("translation_pairs_k3.tsv", sep="\t", header=None, names=["source", "target"])
def tokenize(seq):
    return seq.strip().split()
class NgramModel:
    def __init__(self):
        self.unigrams = Counter()
        self.bigrams = defaultdict(Counter)
        self.total = 0

    def train(self, sequences):
        for seq in sequences:
            tokens = ["<s>"] + tokenize(seq) + ["</s>"]
            for i in range(len(tokens)):
                self.unigrams[tokens[i]] += 1
                self.total += 1
                if i > 0:
                    self.bigrams[tokens[i-1]][tokens[i]] += 1

    def predict_next(self, prev_token):
        if prev_token in self.bigrams:
            return self.bigrams[prev_token].most_common(1)[0][0]
        else:
            return self.unigrams.most_common(1)[0][0]

    def generate(self, max_len=50):
        result = []
        prev = "<s>"
        for _ in range(max_len):
            nxt = self.predict_next(prev)
            if nxt == "</s>":
                break
            result.append(nxt)
            prev = nxt
        return result
data = list(zip(df["source"], df["target"]))
random.shuffle(data)
split = int(len(data) * 0.8)
train_data = data[:split]
test_data = data[split:]

model = NgramModel()
model.train([target for _, target in train_data])
def evaluate(model, test_set):
    total, correct = 0, 0
    for _, target in test_set:
        true_tokens = tokenize(target)
        pred_tokens = model.generate(max_len=len(true_tokens))
        for t, p in zip(true_tokens, pred_tokens):
            total += 1
            if t == p:
                correct += 1
    return correct / total if total > 0 else 0
accuracy = evaluate(model, test_data)
print(f"Token-level Accuracy: {accuracy:.4f}")

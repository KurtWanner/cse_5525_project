import pandas as pd

df = pd.read_csv("eval_5mer.tsv", sep="\t", header=None)

df.columns = ["source", "predicted"]
with open("source_seqs.fasta", "w") as src_fasta, \
     open("predicted_seqs.fasta", "w") as pred_fasta:
    
    for idx, row in df.iterrows():
        src_fasta.write(f">source_{idx}\n{row['source']}\n")
        pred_fasta.write(f">predicted_{idx}\n{row['predicted']}\n")

print("Fasta files generated successfully: source_seqs.fasta, predicted_seqs.fasta")

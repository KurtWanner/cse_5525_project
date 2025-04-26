from Bio import SeqIO
import csv


caroli_fa = "/users/PAS2177/liu9756/Gene Translator/Pre_processing_RNA_seq/mus_caroli_gene_seqs.fa"
pahari_fa = "/users/PAS2177/liu9756/Gene Translator/Pre_processing_RNA_seq/mus_pahari_gene_seqs.fa"
k = 3
output_file = "/users/PAS2177/liu9756/Gene Translator/Pre_processing_RNA_seq/translation_pairs_k3.tsv"

def load_gene_seqs(fasta_path):
    gene_to_seq = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        gene_id = record.id
        seq = str(record.seq).upper()
        if len(seq) >= k:
            gene_to_seq[gene_id] = seq
    return gene_to_seq

caroli_seqs = load_gene_seqs(caroli_fa)
pahari_seqs = load_gene_seqs(pahari_fa)

print(f"Caroli genes loaded: {len(caroli_seqs)}")
print(f"Pahari genes loaded: {len(pahari_seqs)}")

shared_genes = set(caroli_seqs.keys()) & set(pahari_seqs.keys())
print(f"Shared genes: {len(shared_genes)}")

def seq_to_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

with open(output_file, "w") as out:
    writer = csv.writer(out, delimiter="\t")
    writer.writerow(["input_kmers", "output_kmers"])

    for gene_id in shared_genes:
        src_kmers = seq_to_kmers(caroli_seqs[gene_id], k)
        tgt_kmers = seq_to_kmers(pahari_seqs[gene_id], k)
        writer.writerow([" ".join(src_kmers), " ".join(tgt_kmers)])

print(f"Translation pairs written to {output_file}")
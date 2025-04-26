from Bio import SeqIO
from collections import defaultdict
import os


#gtf_path = "/users/PAS2177/liu9756/Gene Translator/Mouse_caroli_dataset/data/GCF_900094665.2/genomic.gtf"
#fna_path = "/users/PAS2177/liu9756/Gene Translator/Mouse_caroli_dataset/data/GCF_900094665.2/GCF_900094665.2_CAROLI_EIJ_v1.1_genomic.fna"
#output_fasta = "/users/PAS2177/liu9756/Gene Translator/Pre_processing_RNA_seq/mus_caroli_gene_seqs.fa"

gtf_path = "/users/PAS2177/liu9756/Gene Translator/mouse_pahari_dataset/ncbi_dataset/data/GCF_900095145.1/genomic.gtf"
fna_path = "/users/PAS2177/liu9756/Gene Translator/mouse_pahari_dataset/ncbi_dataset/data/GCF_900095145.1/GCF_900095145.1_PAHARI_EIJ_v1.1_genomic.fna"
output_fasta = "/users/PAS2177/liu9756/Gene Translator/Pre_processing_RNA_seq/mus_pahari_gene_seqs.fa"

gene_coords = {}
with open(gtf_path) as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        if len(fields) < 9:
            continue
        if fields[2] != "gene":
            continue

        chrom = fields[0]
        start = int(fields[3])
        end = int(fields[4])
        strand = fields[6]

        attributes = fields[8]
        gene_id = ""
        if 'gene_id "' in attributes:
            gene_id = attributes.split('gene_id "')[1].split('"')[0]
        
        if gene_id:
            gene_coords[gene_id] = (chrom, start, end, strand)

print(f"Extracted {len(gene_coords)} gene coordinates from GTF")

chrom_seqs = SeqIO.to_dict(SeqIO.parse(fna_path, "fasta"))

with open(output_fasta, "w") as out:
    for gene_id, (chrom, start, end, strand) in gene_coords.items():
        if chrom not in chrom_seqs:
            continue
        seq = chrom_seqs[chrom].seq[start-1:end]  # 1-based in GTF
        if strand == "-":
            seq = seq.reverse_complement()
        out.write(f">{gene_id}\n{seq}\n")

print(f"Gene sequences saved to {output_fasta}")

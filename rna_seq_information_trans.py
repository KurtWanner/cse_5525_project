from Bio.Blast import NCBIWWW, NCBIXML

def jaccard(s1, s2):
    union = []
    conj = []

    for x in s1:
        if x not in union:
            union.append(x)
        if x in s2 and x not in conj:
            conj.append(x)

    for x in s2:
        if x not in union:
            union.append(x)

    return len(conj) * 1.0 / len(union)


def soren_dice(s1, s2):
    union = []
    conj = []

    for x in s1:
        if x not in union:
            union.append(x)
        if x in s2 and x not in conj:
            conj.append(x)

    for x in s2:
        if x not in union:
            union.append(x)

    return (2.0 * len(conj)) / (len(s1) + len(s2))

def eval_rna(rna1, rna2):
    g1 = blast_rna_seq(rna1)
    g2 = blast_rna_seq(rna2)

    return (soren_dice(g1, g2), jaccard(g1, g2))

def blast_rna_seq(rna_seq):
    rna_seq = rna_seq.upper().replace(" ", "").replace("\n", "").replace("T", "U")
    result_handle = NCBIWWW.qblast("blastn", "nt", rna_seq)
    blast_record = NCBIXML.read(result_handle)
    
    results = []
    for alignment in blast_record.alignments:
        results.append(alignment.title)
    
    return results

def translate_rna_to_protein(rna_seq):
    codon_table = {
        "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
        "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
        "UAU": "Y", "UAC": "Y", "UAA": "Stop", "UAG": "Stop",
        "UGU": "C", "UGC": "C", "UGA": "Stop", "UGG": "W",
        "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
        "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
        "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
        "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G"
    }
    rna_seq = rna_seq.upper().replace(" ", "").replace("\n", "").replace("T", "U")
    
    protein = ""
    for i in range(0, len(rna_seq), 3):
        codon = rna_seq[i:i+3]
        if len(codon) < 3:
            break
        amino_acid = codon_table.get(codon, "")
        if amino_acid == "Stop":
            break
        if amino_acid == "":
            raise ValueError(f"illegal RNA codon: {codon}")
        protein += amino_acid
    return protein

def evaluation():
    with open('results/eval_5mer.tsv', 'r') as f:
        data = [line.split('\t') for line in f.readlines()]

    for x in data:
        s1 = x[0]
        s2 = x[1]
        g1 = blast_rna_seq(s1)
        g2 = blast_rna_seq(s2)
        print(soren_dice(g1, g2), jaccard(g1, g2))


if __name__ == "__main__":
    evaluation()
    sample_rna = "AUGGCCAUGGCGCCCAGAACUGAGAUCAAUAGUACCCGUAUUAACGGGUGA"
    try:
        blast_results = blast_rna_seq(sample_rna)
        print("\nCorresponding gene：")
        for idx, item in enumerate(blast_results, start=1):
            print(f"{idx}. {item}")
    except Exception as e:
        print("BLAST ERROR：", e)

    try:
        protein = translate_rna_to_protein(sample_rna)
        print("Protine Sequence:")
        print(protein)
    except ValueError as ve:
        print(ve)

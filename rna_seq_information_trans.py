from Bio.Blast import NCBIWWW, NCBIXML

def blast_rna_seq(rna_seq):
    rna_seq = rna_seq.upper().replace(" ", "").replace("\n", "")
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
    rna_seq = rna_seq.upper().replace(" ", "").replace("\n", "")
    
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


if __name__ == "__main__":
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

import pandas as pd

def compute_average_identity(blast_file):
    # 读取blast输出
    df = pd.read_csv(blast_file, sep="\t", header=None)
    df.columns = [
        "qseqid", "sseqid", "pident", "length",
        "mismatch", "gapopen", "qstart", "qend",
        "sstart", "send", "evalue", "bitscore"
    ]

    # 每个query取第一条最好的hit
    best_hits = df.groupby("qseqid").first().reset_index()

    # 统计
    avg_identity = best_hits["pident"].mean()
    avg_bitscore = best_hits["bitscore"].mean()

    return avg_identity, avg_bitscore

def main():
    # 输入你的blast结果路径
    source_file = "source_blast.out"
    predicted_file = "predicted_blast.out"

    # 计算平均 identity 和 bitscore
    source_identity, source_bitscore = compute_average_identity(source_file)
    predicted_identity, predicted_bitscore = compute_average_identity(predicted_file)

    # 输出
    print("\n🔹 BLAST Summary Results 🔹\n")
    print(f"{'Type':<15} {'Avg Identity (%)':<20} {'Avg Bitscore':<15}")
    print(f"{'-'*50}")
    print(f"{'Source':<15} {source_identity:.2f}%{'':<10} {source_bitscore:.2f}")
    print(f"{'Predicted':<15} {predicted_identity:.2f}%{'':<10} {predicted_bitscore:.2f}")
    print()

    # 总结
    if predicted_identity > source_identity:
        print("✅ Model translation improves biological similarity (higher identity %).")
    else:
        print("⚠️  Model translation does NOT improve biological similarity.")

if __name__ == "__main__":
    main()

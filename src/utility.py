def print_results(results):

    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    for dataset, m in results.items():
        print(f"\n{dataset.upper()}")
        print("-" * 40)
        print(f"Samples      : {m['Samples']}")
        print(f"Threshold    : {m['Threshold']:.6f}")
        print()
        print(f"TP           : {m['TP']}")
        print(f"FP           : {m['FP']}")
        print(f"TN           : {m['TN']}")
        print(f"FN           : {m['FN']}")
        print()
        print(f"Accuracy     : {m['Accuracy']:.4f}")
        print(f"Precision    : {m['Precision']:.4f}")
        print(f"Recall       : {m['Recall']:.4f}")
        print(f"F1 Score     : {m['F1']:.4f}")
        print(f"Cohen Kappa  : {m['Cohen Kappa']:.4f}")
        print(f"AUROC        : {m['AUROC']:.4f}")
        print(f"AUPRC        : {m['AUPRC']:.4f}")

    print("=" * 80)
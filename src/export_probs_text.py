
"""
Standardize a CSV of text/tabular probabilities into (id,label,prob).

This does NOT train a model. It simply reads your existing CSV and
renames/selects the right columns for fusion.

Example:
python src/export_probs_text.py \
    --in_csv data/.csv/metrics.csv \
    --out_csv .csv/text_probs.csv \
    --id_col id --label_col label --prob_col prob_text

If your file already has columns named (id,label,prob), you can omit the *_col flags.
"""

import os, argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv",  required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--id_col",     default="id")
    ap.add_argument("--label_col",  default="label")
    ap.add_argument("--prob_col",   default="prob")  
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    df = pd.read_csv(args.in_csv)
    for col in (args.id_col, args.label_col, args.prob_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {args.in_csv}")

    out = df[[args.id_col, args.label_col, args.prob_col]].rename(
        columns={args.id_col: "id", args.label_col: "label", args.prob_col: "prob"}
    )
    out.to_csv(args.out_csv, index=False)
    print(f"[text] wrote {args.out_csv} ({len(out)} rows)")

if __name__ == "__main__":
    main()

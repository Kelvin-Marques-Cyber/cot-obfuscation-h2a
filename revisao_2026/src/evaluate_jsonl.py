"""Avalia novas gerações previamente coletadas; não chama provedores de modelos.

Uso: python3 src/evaluate_jsonl.py generations.jsonl results/new_evaluation.csv
Contrato de entrada por linha:
item_id, model_id, model_revision, condition, train_seed, dataset_split,
prompt, gold, generation, finish_reason, input_tokens, output_tokens.
"""
import csv
import json
import sys
from pathlib import Path
from instruments import parse_final, strict_grade

REQUIRED={"item_id","model_id","model_revision","condition","train_seed",
          "dataset_split","prompt","gold","generation","finish_reason",
          "input_tokens","output_tokens"}


def evaluate(source, destination):
    rows=[]
    seen=set()
    for i,line in enumerate(Path(source).read_text().splitlines(),1):
        if not line.strip():
            continue
        row=json.loads(line)
        if missing:=REQUIRED-set(row):
            raise ValueError(f"Linha {i}: faltam campos {sorted(missing)}")
        identity=(row["item_id"],row["model_id"],row["model_revision"],
                  row["condition"],row["train_seed"])
        if identity in seen:
            raise ValueError(f"Linha {i}: avaliação duplicada")
        seen.add(identity)
        final=parse_final(row["generation"])
        rows.append(dict(item_id=row["item_id"],model_id=row["model_id"],
                         condition=row["condition"],train_seed=row["train_seed"],
                         format_valid=int(final is not None),
                         accuracy=int(strict_grade(row["generation"],row["gold"])),
                         truncated=int(row["finish_reason"]=="length")))
    if not rows:
        raise ValueError("Arquivo de gerações vazio")
    with Path(destination).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__=="__main__":
    if len(sys.argv)!=3:
        raise SystemExit(__doc__)
    print(f"{len(evaluate(sys.argv[1],sys.argv[2]))} gerações avaliadas.")

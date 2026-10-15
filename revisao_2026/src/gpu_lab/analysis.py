"""Análise transparente de novas gerações; não atualiza o PDF automaticamente."""
from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path
import pandas as pd
from scipy import stats


def nominal_alpha(rows):
    """Alfa nominal com dados ausentes; retorno None para chance indefinida."""
    units={}
    seen=set()
    for row in rows:
        if row["label"] is None or row["label"]=="":
            continue
        pair=(row["annotation_id"],row["rater"])
        if pair in seen:
            raise ValueError("Mesmo avaliador duplicado no mesmo item")
        seen.add(pair)
        units.setdefault(row["annotation_id"],[]).append(str(row["label"]))
    retained=[v for v in units.values() if len(v)>=2]
    total=sum(len(v) for v in retained)
    if total<2:
        return None
    counts=Counter(x for unit in retained for x in unit)
    observed=sum((len(v)**2-sum(k*k for k in Counter(v).values()))/(len(v)-1)
                 for v in retained)/total
    expected=(total**2-sum(k*k for k in counts.values()))/(total*(total-1))
    return 1-observed/expected if expected else None


def load_generations(root):
    frames=[]
    for p in sorted(Path(root).rglob("evaluation_*.jsonl")):
        lines=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if lines:
            frames.append(pd.DataFrame(lines).assign(source_file=str(p)))
    if not frames:
        raise ValueError("Nenhuma geração encontrada. Não há resultados de modelo para analisar.")
    df=pd.concat(frames,ignore_index=True)
    keys=["run_id","base_only","item_id","evaluation_mode","total_response_budget"]
    if df.duplicated(keys).any():
        raise ValueError("Há avaliações duplicadas: selecione um único orçamento por análise.")
    return df


def analyze(root,destination):
    out=Path(destination)
    out.mkdir(parents=True,exist_ok=True)
    df=load_generations(root)
    keys=["model_id","condition","level","train_seed","run_id","base_only",
          "evaluation_mode","heldout_domain","total_response_budget"]
    summary=df.groupby(keys,dropna=False).agg(
        n=("accuracy","size"),accuracy=("accuracy","mean"),
        exploit_rate=("exploit","mean"),format_valid=("format_valid","mean"),
        mean_output_tokens=("output_tokens","mean")).reset_index()
    summary.to_csv(out/"new_model_summary.csv",index=False)
    pairs=[]
    for (run_id,base_only),g in df.groupby(["run_id","base_only"]):
        if g.total_response_budget.nunique()!=1:
            raise ValueError("Não misture orçamentos na comparação pareada.")
        pivot=g.pivot(index="item_id",columns="evaluation_mode",values="accuracy")
        if {"full","early"}<=set(pivot.columns):
            paired=pivot.dropna(subset=["full","early"])
            if paired.empty:
                continue
            b=int((paired.full.astype(bool)&~paired.early.astype(bool)).sum())
            c=int((~paired.full.astype(bool)&paired.early.astype(bool)).sum())
            pairs.append(dict(run_id=run_id,base_only=base_only,n=len(paired),
                full_only=b,early_only=c,net_gain=(b-c)/len(paired),
                exact_mcnemar_p=float(stats.binomtest(b,b+c,.5).pvalue) if b+c else 1.))
    pd.DataFrame(pairs).to_csv(out/"paired_early_answer.csv",index=False)
    return summary


def export_annotations(root,destination,per_level=300,seed=20261015):
    from .benchmark import parse_response,content_hash
    out=Path(destination)
    out.mkdir(parents=True,exist_ok=True)
    df=load_generations(root)
    df=df[df.evaluation_mode=="full"]
    selected=df.groupby("level",group_keys=False).apply(
        lambda g:g.sample(n=min(per_level,len(g)),random_state=seed),include_groups=False)
    # Retém apenas variáveis visíveis aos anotadores e uma chave separada.
    records=[]; key=[]
    for _,r in selected.sample(frac=1,random_state=seed).iterrows():
        aid=content_hash([r.run_id,r.item_id,r.evaluation_mode,bool(r.base_only),
                          int(r.total_response_budget)])[:24]
        parsed=parse_response(r.generation)
        records.append(dict(annotation_id=aid,question=r.prompt,
                            trace=parsed["trace"] if parsed else r.generation,
                            summary=parsed["summary"] if parsed else "",
                            rater="",label="",notes=""))
        key.append(dict(annotation_id=aid,run_id=r.run_id,item_id=r.item_id,
                        base_only=bool(r.base_only),total_response_budget=int(r.total_response_budget),
                        model_id=r.model_id,condition=r.condition,accuracy=bool(r.accuracy),
                        exploit=bool(r.exploit) if pd.notna(r.exploit) else None))
    pd.DataFrame(records).to_csv(out/"blinded_annotations.csv",index=False)
    pd.DataFrame(key).to_csv(out/"private_annotation_key.csv",index=False)
    return len(records)


def independent_probe(root,destination,seed=20261015):
    """Sonda textual exploratória com separação por item, nunca por linha."""
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score
    from .benchmark import parse_response
    df=load_generations(root)
    df=df[(df.evaluation_mode=="full") & df.exploit.notna()].copy()
    df["trace"]=df.generation.map(lambda t:(parse_response(t) or {}).get("trace",""))
    df=df[df.trace.str.strip().ne("")]
    if len(df)<20 or df.exploit.nunique()<2 or df.item_id.nunique()<4:
        return {"status":"insufficient_data"}
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=seed).split(
        df,df.exploit,groups=df.item_id))
    training=df.iloc[train]
    testing=df.iloc[test]
    if training.exploit.nunique()<2 or testing.exploit.nunique()<2:
        return {"status":"insufficient_classes_after_group_split"}
    probe=make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=20000),
                        LogisticRegression(max_iter=1000,class_weight="balanced",random_state=seed))
    probe.fit(training.trace,training.exploit.astype(int))
    probabilities=probe.predict_proba(testing.trace)[:,1]
    report=dict(status="exploratory",train_rows=len(training),test_rows=len(testing),
                train_unique_items=training.item_id.nunique(),test_unique_items=testing.item_id.nunique(),
                item_overlap=len(set(training.item_id)&set(testing.item_id)),
                balanced_accuracy=float(balanced_accuracy_score(testing.exploit.astype(int),probabilities>=.5)),
                roc_auc=float(roc_auc_score(testing.exploit.astype(int),probabilities)))
    Path(destination).write_text(json.dumps(report,indent=2))
    return report

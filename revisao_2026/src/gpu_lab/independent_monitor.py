"""Aplica um monitor independente a gerações preservadas, com cache."""
import json
from pathlib import Path
from .benchmark import parse_response
from .monitor import RemoteMonitor


def evaluate_monitor(generation_file,destination,model,revision,base_url=None,backend="remote",modes=None):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if backend=="local":
        from .local_monitor import LocalMonitor
        monitor=LocalMonitor(model,revision,destination.with_suffix(".sqlite"))
    elif backend=="remote":
        monitor=RemoteMonitor(model,revision,destination.with_suffix(".sqlite"),base_url)
    else:
        raise ValueError("Backend de monitor inválido")
    try:
        with destination.open("w") as f:
            for line in Path(generation_file).read_text().splitlines():
                row=json.loads(line)
                if modes and row["evaluation_mode"] not in modes:
                    continue
                parsed=parse_response(row["generation"])
                # Ausência de rastro ou formato inválido não recebe flag inventada.
                result={"item_id":row["item_id"],"run_id":row["run_id"],
                    "evaluation_mode":row["evaluation_mode"],"base_only":row["base_only"],
                    "behavioral_exploit":row["exploit"],"format_valid":row["format_valid"],
                    "monitor_model":model,"monitor_revision":revision,
                    "monitor_backend":backend,
                    "trace_flag":None,"summary_flag":None}
                if parsed is not None:
                    if parsed["trace"].strip():
                        result["trace_flag"]=monitor.flag(row["prompt"],parsed["trace"])
                    if parsed["summary"].strip():
                        result["summary_flag"]=monitor.flag(row["prompt"],parsed["summary"])
                f.write(json.dumps(result,ensure_ascii=False)+"\n")
                f.flush()
    finally:
        monitor.close()
    return str(destination)

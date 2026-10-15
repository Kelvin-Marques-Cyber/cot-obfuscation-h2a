"""Aplica um monitor independente a gerações preservadas, com cache."""
import json
from pathlib import Path
from .benchmark import parse_response
from .monitor import RemoteMonitor


def evaluate_monitor(generation_file,destination,model,revision,base_url=None):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    monitor=RemoteMonitor(model,revision,destination.with_suffix(".sqlite"),base_url)
    try:
        with destination.open("w") as f:
            for line in Path(generation_file).read_text().splitlines():
                row=json.loads(line)
                parsed=parse_response(row["generation"])
                # Ausência de rastro ou formato inválido não recebe flag inventada.
                result={"item_id":row["item_id"],"run_id":row["run_id"],
                    "evaluation_mode":row["evaluation_mode"],"base_only":row["base_only"],
                    "behavioral_exploit":row["exploit"],"format_valid":row["format_valid"],
                    "monitor_model":model,"monitor_revision":revision,
                    "trace_flag":None,"summary_flag":None}
                if parsed is not None:
                    result["trace_flag"]=monitor.flag(row["prompt"],parsed["trace"])
                    result["summary_flag"]=monitor.flag(row["prompt"],parsed["summary"])
                f.write(json.dumps(result,ensure_ascii=False)+"\n")
                f.flush()
    finally:
        monitor.close()
    return str(destination)

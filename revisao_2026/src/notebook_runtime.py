"""Execução de comandos com saída visível e registro persistente no notebook."""
from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import subprocess


def run_logged(command, log_path, *, cwd=None, env=None):
    """Transmite stdout/stderr juntos e inclui o erro real na exceção.

    O log é sobrescrito a cada execução da etapa. As últimas linhas também
    aparecem na exceção, pois alguns clientes só exibem o traceback da célula.
    """
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=40)
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            [os.fspath(arg) for arg in command], cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                tail.append(line)
            returncode = process.wait()
    if returncode:
        raise RuntimeError(
            f"A etapa terminou com código {returncode}. Log completo: {log_path}\n"
            + "".join(tail)
        )
    return subprocess.CompletedProcess(command, returncode)

"""Executa as verificações locais do repositório e registra a procedência.

Para cada verificação: comando, diretório, duração, código de saída, final da
saída e SHA256 dos arquivos produzidos. Registra também versões e hardware.
Os testes são de software; nenhum deles produz evidência científica.

Uso: python revisao_2026/src/executar_validacao.py --out revisao_2026/results/execucoes_locais.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    out = dict(python=sys.version.split()[0], platform=platform.platform())
    for name in ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'datasets', 'numpy', 'pandas', 'scipy',
                 'matplotlib', 'docx'):
        try:
            module = __import__(name)
            out[name] = getattr(module, '__version__', 'instalado')
        except Exception:
            out[name] = None
    try:
        cpu = [l for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')][0]
        out['cpu'] = cpu.split(':', 1)[1].strip()
        out['cpu_count'] = os.cpu_count()
        mem = [l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal')][0]
        out['ram_gib'] = round(int(mem.split()[1]) / 1024 ** 2, 1)
    except (OSError, IndexError):
        pass
    try:
        out['pandoc'] = subprocess.run(['pandoc', '--version'], capture_output=True, text=True).stdout.split('\n')[0]
        out['libreoffice'] = subprocess.run(['soffice', '--version'], capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return out


def clean(value):
    """Remove caminhos locais (interpretador, repositório, temporários) dos registros publicados."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if not isinstance(value, str):
        return value
    value = value.replace(sys.executable, 'python').replace(str(REPO), '.')
    return re.sub(r'(?:/tmp|' + re.escape(tempfile.gettempdir()) + r')/[^\s\'",)]*', '<temporário>', value)


def run(name, command, cwd, produced=(), env=None, timeout=3600):
    started = time.time()
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, env=env, timeout=timeout)
    text = result.stdout + result.stderr
    summary = [line for line in text.splitlines() if line.startswith(('Ran ', 'OK', 'FAILED'))]
    entry = dict(name=name, command=clean(' '.join(map(str, command))), cwd=str(Path(cwd).relative_to(REPO)),
                 exit_code=result.returncode, seconds=round(time.time() - started, 1), summary=summary[-3:],
                 tail=clean(text.strip().splitlines()[-6:]),
                 produced={str(Path(p).relative_to(REPO)): sha(p) for p in produced if Path(p).exists()})
    print(f'[{entry["exit_code"]}] {name} ({entry["seconds"]} s) {" ".join(summary[-2:])}', flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPO / 'revisao_2026' / 'results' / 'execucoes_locais.json'))
    parser.add_argument('--skip-slow', action='store_true')
    args = parser.parse_args()
    py = sys.executable
    lab, v3, v4 = REPO / 'revisao_2026', REPO / 'experimento_cientifico_v3', REPO / 'experimento_cientifico_v4'
    entries = []
    entries.append(run('Laboratório de reanálise: testes unitários', [py, '-m', 'unittest', 'discover', '-s', 'tests'], lab))
    entries.append(run('Reanálise histórica (reanalyse.py)', [py, 'src/reanalyse.py'], lab,
                       produced=sorted((lab / 'results').glob('*.csv')) + [lab / 'results' / 'report.json']))
    entries.append(run('Estudo v3.2: testes de contrato e armazenamento', [py, '-m', 'unittest', 'discover'], v3))
    entries.append(run('Estudo v4: contratos, armazenamento e sessão',
                       [py, '-m', 'unittest', 'tests.test_contratos', 'tests.test_colab'], v4))
    if not args.skip_slow:
        entries.append(run('Estudo v4: integração com modelos mínimos de pesos aleatórios (CPU)',
                           [py, '-m', 'unittest', 'tests.test_integracao_minima'], v4))
    with tempfile.TemporaryDirectory() as temp:
        report = Path(temp) / 'notebooks.json'
        entries.append(run('Estudo v4: cinco notebooks com Colab simulado (montagem e download falhando)',
                           [py, 'tests/integracao_notebooks.py', '--report', str(report)], v4,
                           env=dict(os.environ, COT_TEST_TMP=temp)))
        if report.exists():
            entries[-1]['report'] = clean(json.loads(report.read_text()))
    data = dict(date=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), environment=versions(),
                git_head=subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True,
                                        text=True).stdout.strip(),
                scientific_evidence=False, entries=entries,
                all_passed=all(e['exit_code'] == 0 for e in entries))
    Path(args.out).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Relatório:', args.out, '| todas aprovadas:', data['all_passed'])
    sys.exit(0 if data['all_passed'] else 1)


if __name__ == '__main__':
    main()

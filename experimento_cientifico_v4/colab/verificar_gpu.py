"""Verificação do ambiente de GPU antes de qualquer download de pesos."""
import importlib.metadata as metadata
import sys

EXPECTED = {'transformers': '4.56.2', 'trl': '0.22.2', 'peft': '0.17.1',
            'accelerate': '1.10.1', 'datasets': '4.3.0'}


def main(min_vram):
    import torch
    if not torch.cuda.is_available():
        sys.exit('CUDA indisponível no ambiente isolado.')
    for package, version in EXPECTED.items():
        found = metadata.version(package)
        if found != version:
            sys.exit(f'Versão divergente de {package}: {found} (esperada {version}).')
    import peft, transformers, trl  # noqa: F401  (importações reais, não apenas metadados)
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1024 ** 3
    if vram < min_vram:
        sys.exit(f'VRAM insuficiente: {vram:.1f} GiB < {min_vram} GiB.')
    a = torch.randn(256, 256, device='cuda', dtype=torch.float16)
    product = a @ a.T
    if not torch.isfinite(product).all():
        sys.exit('Teste fp16 em CUDA produziu valores não finitos.')
    print(f'GPU verificada: {props.name}, {vram:.1f} GiB, capacidade {props.major}.{props.minor}, '
          f'torch {torch.__version__}, CUDA {torch.version.cuda}')


if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 14.0)

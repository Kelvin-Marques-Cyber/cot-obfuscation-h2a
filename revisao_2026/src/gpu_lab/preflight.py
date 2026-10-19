"""Verificações do ambiente de texto antes de baixar pesos e iniciar treinos."""
import importlib.metadata as metadata
import os


def text_only_compat():
    if os.environ.get("COT_TEXT_ONLY") == "1":
        import importlib.util
        if importlib.util.find_spec("transformers") is None:
            return  # A auditoria CPU não instala bibliotecas de treino.
        # TRL/Transformers estão fixados. O laboratório não usa áudio nem visão;
        # extensões opcionais herdadas do Colab não precisam entrar no processo.
        from transformers.utils import import_utils
        import_utils._torchvision_available = False
        import_utils._librosa_available = False
        if os.environ.get("COT_PROFILE")=="quick":
            # PEFT 0.17.1 consulta seu próprio detector, independente do
            # Transformers. O LoRA fp16 não deve importar um bnb herdado.
            from peft.tuners.lora import model as lora_model
            lora_model.is_bnb_available = lambda: False
            lora_model.is_bnb_4bit_available = lambda: False


def main():
    text_only_compat()
    import torch
    if os.environ.get("COT_PROFILE") != "quick":
        import bitsandbytes
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig
    from trl import SFTTrainer, GRPOTrainer
    import pandas, scipy, sklearn
    assert torch.cuda.is_available(), "GPU CUDA ausente"
    for package, expected in {"transformers":"4.56.2","trl":"0.22.2","peft":"0.17.1"}.items():
        assert metadata.version(package)==expected, f"Versão divergente: {package}"
    print("Importações GPU verificadas:",torch.cuda.get_device_name(0),
          "| Perfil:",os.environ.get("COT_PROFILE","custom"),
          "| VRAM total:",round(torch.cuda.get_device_properties(0).total_memory/1024**3,1),"GiB")


if __name__ == "__main__":
    main()

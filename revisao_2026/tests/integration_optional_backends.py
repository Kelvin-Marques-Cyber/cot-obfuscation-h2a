"""LoRA fp16 deve funcionar mesmo com um bnb global incompatível.

PYTHONPATH=src COT_TEXT_ONLY=1 COT_PROFILE=quick python tests/integration_optional_backends.py
"""
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    if os.environ.get("COT_PROFILE")!="quick" or os.environ.get("COT_TEXT_ONLY")!="1":
        raise RuntimeError("Execute com COT_PROFILE=quick e COT_TEXT_ONLY=1")
    with tempfile.TemporaryDirectory() as tmp:
        package=Path(tmp)/"bitsandbytes"
        package.mkdir()
        (package/"__init__.py").write_text("raise RuntimeError('BITSANDBYTES_HERDADO_INCOMPATIVEL')\n")
        sys.path.insert(0,tmp)
        try:
            from gpu_lab.preflight import text_only_compat
            text_only_compat()
            from transformers import Qwen2Config,Qwen2ForCausalLM
            from peft import LoraConfig,get_peft_model
            model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1))
            adapted=get_peft_model(model,LoraConfig(r=2,task_type="CAUSAL_LM",target_modules="all-linear"))
            assert any(p.requires_grad for p in adapted.parameters())
            assert "bitsandbytes" not in sys.modules
            print(json.dumps(dict(status="passed",device="cpu",broken_inherited_bitsandbytes_ignored=True)))
        finally:
            sys.path.remove(tmp)


if __name__=="__main__":
    main()

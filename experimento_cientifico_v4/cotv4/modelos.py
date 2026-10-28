"""Carregamento de modelos, modelo de conversa e geração gulosa."""
from __future__ import annotations

import time

REVISIONS = {
    'Qwen/Qwen2.5-1.5B-Instruct': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
    'Qwen/Qwen3-1.7B': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
    'Qwen/Qwen2.5-0.5B-Instruct': '7ae557604adf67be50417f59c2c2f167def9a775',
    'HuggingFaceTB/SmolLM2-135M-Instruct': '12fd25f77366fa6b3b4b768ec3050bf629380bac',
}


def local_revision(model_id, revision):
    """Modelos em diretório local (somente testes de software) não têm revisão do Hub."""
    from pathlib import Path
    return None if Path(model_id).is_dir() else revision


def load_model(model_id, revision, device='cpu', dtype='float32', adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA indisponível: selecione um ambiente com GPU.')
    revision = local_revision(model_id, revision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=getattr(torch, dtype),
        device_map={'': 0 if device == 'cuda' else 'cpu'}, attn_implementation='sdpa')
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def chat(tokenizer, messages):
    """Modelo de conversa oficial; Qwen3 sem modo de pensamento."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def generate(model, tokenizer, prompt, budget):
    import torch
    encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
    encoded.pop('token_type_ids', None)
    limit = getattr(model.config, 'max_position_embeddings', 10 ** 9)
    if encoded['input_ids'].shape[1] + budget > limit:
        raise ValueError('Contexto excede a janela do modelo; não será truncado.')
    started = time.perf_counter()
    with torch.inference_mode():
        tokens = model.generate(**encoded, max_new_tokens=budget, do_sample=False,
                                temperature=None, top_p=None, top_k=None,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id)
    new = tokens[0, encoded['input_ids'].shape[1]:]
    ids = new.tolist()
    return dict(generation=tokenizer.decode(new, skip_special_tokens=True),
                input_tokens=int(encoded['input_ids'].shape[1]), output_tokens=len(ids),
                finish_reason='eos' if ids and ids[-1] == tokenizer.eos_token_id else 'length',
                seconds=time.perf_counter() - started)


def generate_batch(model, tokenizer, prompts, budget):
    """Geração gulosa em lote com preenchimento à esquerda (somente calibração exploratória)."""
    import torch
    encoded = tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
    encoded.pop('token_type_ids', None)
    with torch.inference_mode():
        tokens = model.generate(**encoded, max_new_tokens=budget, do_sample=False, temperature=None,
                                top_p=None, top_k=None, eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id)
    outputs = []
    width = encoded['input_ids'].shape[1]
    for row in tokens:
        ids = row[width:].tolist()
        while ids and ids[-1] == tokenizer.pad_token_id and tokenizer.pad_token_id != tokenizer.eos_token_id:
            ids.pop()
        finished = tokenizer.eos_token_id in ids
        if finished:
            ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
        outputs.append(dict(generation=tokenizer.decode(ids, skip_special_tokens=True), output_tokens=len(ids),
                            finish_reason='eos' if finished else 'length'))
    return outputs

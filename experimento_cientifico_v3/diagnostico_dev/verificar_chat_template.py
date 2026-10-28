import sys, json
sys.path.insert(0, sys.argv[1] + '/revisao_2026/src')
from transformers import AutoTokenizer
from gpu_lab.train import chat
from gpu_lab.benchmark import SYSTEM, render_prompt, create_dataset
out = {}
for model, rev, system, user in [
    ('Qwen/Qwen2.5-1.5B-Instruct', '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
     'Resolva o problema. Termine com uma única tag <answer>NUMERO</answer>.', 'Calcule 3678 + 6943.'),
    ('Qwen/Qwen3-1.7B', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e', SYSTEM,
     render_prompt([r for r in create_dataset(60) if r['split'] == 'dev' and r['level'] == 'low'][0]))]:
    tok = AutoTokenizer.from_pretrained(model, revision=rev)
    prompt = chat(tok, [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}])
    with_special = tok(prompt)['input_ids']
    without = tok(prompt, add_special_tokens=False)['input_ids']
    out[model] = dict(rendered=prompt, tokens=len(with_special),
                      add_special_tokens_changes_ids=with_special != without,
                      eos_token=tok.eos_token, eos_id=tok.eos_token_id, pad_token=tok.pad_token,
                      ends_with=prompt[-40:])
print(json.dumps(out, indent=1, ensure_ascii=False))

"""Train and evaluate the Week 9 router on one real local language model."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import random
import statistics
import time

from triage.evaluate import load_benchmark, p95, score_one
from triage.nodes.prompt_node import build_system_prompt
from triage.nodes.system_one import SystemOne, load_train_rows
from triage.paths import ROUTING_DATA
from triage.spec import ROUTE_INTENTS, URGENCIES, canonical, validate_triage

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent.parent / 'quiz2' / 'Output'
MODEL = 'Qwen/Qwen3-1.7B'
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
MODEL_DIR = ROOT.parent.parent / '.runtime' / 'qwen3-1.7b'
SEED = 42
GPU_USD_PER_HOUR = 0.50
CPU_USD_PER_HOUR = 0.04


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def messages(text):
    return [{'role': 'system', 'content': build_system_prompt(True)}, {'role': 'user', 'content': text}]


def load_model():
    import torch
    import truststore
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    truststore.inject_into_ssl()
    set_seed(SEED)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 support is required for the recorded configuration')
    from huggingface_hub import snapshot_download
    source = snapshot_download(MODEL, revision=REVISION, local_dir=MODEL_DIR,
                               allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        device_map={'': 0}, attn_implementation='sdpa', use_safetensors=True,
    )
    return model, tokenizer


def encoded_prompt(tokenizer, text):
    return tokenizer.apply_chat_template(messages(text), tokenize=True, add_generation_prompt=True, enable_thinking=False)


def split_examples():
    rows = load_train_rows()
    cases = load_benchmark()
    benchmark_inputs = {case['input'].strip() for case in cases}
    by_intent = {}
    for row in rows:
        if row['text'].strip() in benchmark_inputs:
            raise ValueError('Exact training/benchmark prompt overlap')
        by_intent.setdefault(row['intent'], []).append(row)
    train, dev = [], []
    rng = random.Random(SEED)
    for intent in sorted(by_intent):
        unique = {row['text']: row for row in by_intent[intent]}
        group = list(unique.values())
        rng.shuffle(group)
        count = max(1, round(len(group) * .10))
        dev.extend(group[:count])
        train.extend(group[count:])
    if {row['text'] for row in train} & {row['text'] for row in dev}:
        raise ValueError('Training and validation prompts overlap')
    rng.shuffle(train)
    return train, dev


def train():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    adapter = OUTPUT / 'adapter'
    if (adapter / 'adapter_model.safetensors').exists():
        raise FileExistsError(f'Existing adapter at {adapter}. Archive it before retraining.')
    model, tokenizer = load_model()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
    model.config.use_cache = False
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    starts = {name: value.detach().float().cpu().clone() for name, value in trainable.items() if 'lora_B' in name}
    train_rows, dev_rows = split_examples()

    def encode(row):
        prefix = encoded_prompt(tokenizer, row['text'])
        answer = canonical(row['route'], row['intent'], row['urgency'], float(row.get('confidence', .92)))
        completion = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        return {'input_ids': prefix + completion, 'labels': [-100] * len(prefix) + completion}

    def collate(rows):
        length = max(len(row['input_ids']) for row in rows)
        inputs = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full_like(inputs, -100)
        mask = torch.zeros_like(inputs)
        for index, row in enumerate(rows):
            size = len(row['input_ids'])
            inputs[index, :size] = torch.tensor(row['input_ids'])
            labels[index, :size] = torch.tensor(row['labels'])
            mask[index, :size] = 1
        return {'input_ids': inputs, 'labels': labels, 'attention_mask': mask}

    config = TrainingArguments(
        output_dir=str(ROOT / 'results' / 'sft_work'), num_train_epochs=2,
        per_device_train_batch_size=4, per_device_eval_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=2e-4, warmup_steps=5,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        save_strategy='no', eval_strategy='epoch', prediction_loss_only=True, logging_steps=5,
        remove_unused_columns=False, report_to='none', seed=SEED, data_seed=SEED, disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=config, train_dataset=[encode(row) for row in train_rows],
                      eval_dataset=[encode(row) for row in dev_rows], data_collator=collate)
    print(f'SFT starts with {len(train_rows)} train and {len(dev_rows)} validation examples', flush=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start
    delta = sum((trainable[name].detach().float().cpu() - before).square().sum().item() for name, before in starts.items()) ** .5
    if delta <= 0 or trainer.state.global_step <= 0:
        raise RuntimeError('Training did not update adapter weights')
    model.peft_config['default'].base_model_name_or_path = MODEL
    model.peft_config['default'].revision = REVISION
    model.save_pretrained(adapter, safe_serialization=True)
    save(OUTPUT / 'llm_training.json', {
        'model': MODEL, 'revision': REVISION, 'method': 'LoRA SFT, assistant completion only',
        'gpu': torch.cuda.get_device_name(0), 'train_examples': len(train_rows), 'dev_examples': len(dev_rows),
        'train_intents': dict(Counter(row['intent'] for row in train_rows)), 'seed': SEED,
        'steps': trainer.state.global_step, 'epochs': trainer.state.epoch, 'seconds': seconds,
        'trainable_parameters': sum(x.numel() for x in trainable.values()), 'lora_B_change_l2': delta,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'history': trainer.state.log_history,
        'train_data_sha256': digest(ROUTING_DATA / 'train_rows.json'),
        'adapter_sha256': digest(adapter / 'adapter_model.safetensors'),
    })


def build_token_trie(tokenizer):
    """Restrict generation to canonical JSON, including valid route/intent pairs."""
    trie = {}
    count, max_tokens = 0, 0
    for route, intents in ROUTE_INTENTS.items():
        for intent, urgency, percentage in product(intents, URGENCIES, range(101)):
            answer = canonical(route, intent, urgency, percentage / 100)
            ids = tokenizer.encode(answer, add_special_tokens=False)
            if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != answer or validate_triage(answer)[0] is None:
                raise ValueError('Invalid canonical branch')
            node = trie
            for token in ids + [tokenizer.eos_token_id]:
                node = node.setdefault(token, {})
            max_tokens = max(max_tokens, len(ids) + 1)
            count += 1
    return trie, count, max_tokens


def benchmark():
    import torch
    from peft import PeftModel
    from transformers import GenerationConfig

    training = json.loads((OUTPUT / 'llm_training.json').read_text(encoding='utf-8'))
    adapter = OUTPUT / 'adapter'
    if digest(adapter / 'adapter_model.safetensors') != training['adapter_sha256']:
        raise ValueError('Training adapter hash does not match')
    cases = load_benchmark()
    model, tokenizer = load_model()
    model.eval()
    trie, candidates, trie_tokens = build_token_trie(tokenizer)
    config = GenerationConfig(do_sample=False, max_new_tokens=96, use_cache=True,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    if trie_tokens > config.max_new_tokens:
        raise ValueError('Token budget is shorter than a valid grammar branch')

    def generate(text, constrained=False):
        prompt = encoded_prompt(tokenizer, text)
        inputs = torch.tensor([prompt], device='cuda')
        kwargs = {}
        if constrained:
            def allowed(_index, tokens):
                node = trie
                for token in tokens[len(prompt):].tolist():
                    node = node[token]
                return list(node)
            kwargs['prefix_allowed_tokens_fn'] = allowed
        with torch.inference_mode():
            output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                    generation_config=config, use_model_defaults=False, **kwargs)
        ids = output[0, len(prompt):].tolist()
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False), len(prompt), len(ids)

    def run(label, inference, hourly_price):
        inference('查詢我的訂單配送進度')  # Exclude a fixed warm-up from timing.
        records = []
        for case in cases:
            tick = time.perf_counter()
            answer, tin, tout = inference(case['input'])
            elapsed = time.perf_counter() - tick
            records.append({'id':case['id'], 'kind':case['kind'], 'raw':answer, 'input_tokens':tin, 'output_tokens':tout,
                            'latency_ms':elapsed * 1000, 'estimated_usd':elapsed / 3600 * hourly_price,
                            **score_one(answer, case['gold'])})
        n = len(records)
        scores = {key:sum(row[key] for row in records)/n for key in ['json_ok','schema_ok','lenient_ok','route_ok','intent_ok','urgency_ok','exact_ok']}
        latencies = [row['latency_ms'] for row in records]
        summary = {'method':label, 'n':n, **scores, 'latency_mean_ms':statistics.mean(latencies),
                   'latency_p95_ms':p95(latencies), 'latency_median_ms':statistics.median(latencies),
                   'cold_start_and_training_excluded': True, 'api_cost_usd':0,
                   'estimated_usd_per_1000':1000*statistics.mean(row['estimated_usd'] for row in records),
                   'by_kind':{kind:{'n': len(sample), 'intent_ok': statistics.mean(row['intent_ok'] for row in sample)}
                              for kind in sorted({row['kind'] for row in records})
                              for sample in [[row for row in records if row['kind']==kind]]}, 'records':records}
        print(label, {key:round(summary[key],4) for key in ['schema_ok','intent_ok','exact_ok','latency_mean_ms']}, flush=True)
        return summary

    results = {'model':MODEL, 'revision':REVISION, 'benchmark_sha256':digest(ROUTING_DATA / 'benchmark.jsonl'),
               'gpu':torch.cuda.get_device_name(0), 'grammar_candidates':candidates, 'grammar_max_tokens':trie_tokens,
               'cost_assumptions_usd_per_hour':{'gpu':GPU_USD_PER_HOUR,'cpu':CPU_USD_PER_HOUR},
               'transport':'local; same frozen text history and OCR transcripts for every method',
               'generation':config.to_dict(), 'training_sha256':digest(OUTPUT/'llm_training.json'), 'methods':{}}
    results['methods']['A'] = run('A Prompt only', generate, GPU_USD_PER_HOUR)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
    results['methods']['B'] = run('B LoRA SFT', generate, GPU_USD_PER_HOUR)
    results['methods']['C'] = run('C LoRA SFT + token constraint', lambda text: generate(text, True), GPU_USD_PER_HOUR)
    train_rows, _ = split_examples()
    system_one = SystemOne.train(train_rows)
    def predict(text):
        return system_one.run(text).raw, None, None
    results['methods']['D'] = run('D System One TF-IDF + LR', predict, CPU_USD_PER_HOUR)
    if not all(row['schema_ok'] for row in results['methods']['C']['records']):
        raise RuntimeError('Constrained output failed schema validation')
    save(OUTPUT / 'abcd_results.json', results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['train','benchmark'])
    {'train':train, 'benchmark':benchmark}[parser.parse_args().stage]()

"""Shared train/evaluate model and prompt construction. No test answers enter prompts."""
from common import MODELS

SYSTEM = "請完成使用者任務，遵守指定的輸出格式。"


def processor_for(kind):
    from transformers import AutoTokenizer,AutoProcessor
    item=MODELS[kind]
    options=dict(revision=item["revision"],cache_dir=item["cache"],local_files_only=True,trust_remote_code=False)
    if kind=="llm":
        tokenizer=AutoTokenizer.from_pretrained(item["model_id"],**options)
        tokenizer.pad_token=tokenizer.eos_token
        return tokenizer
    processor=AutoProcessor.from_pretrained(item["model_id"],**options,use_fast=False,
        do_image_splitting=False,size={"longest_edge":384})
    processor.image_processor.do_image_splitting=False
    processor.image_processor.size={"longest_edge":384}
    return processor


def render_prompt(kind,processor,prompt):
    if kind=="llm":
        return processor.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return processor.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}],
        tokenize=False,add_generation_prompt=True)


def load_model(kind,training=False,adapter=None):
    import torch
    from transformers import AutoModelForCausalLM,AutoModelForImageTextToText,BitsAndBytesConfig
    from common import require
    item=MODELS[kind]
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"CUDA BF16 required")
    options=dict(revision=item["revision"],cache_dir=item["cache"],local_files_only=True,
        torch_dtype=torch.bfloat16,device_map={"":0},attn_implementation="sdpa",trust_remote_code=False,
        low_cpu_mem_usage=True)
    if kind=="llm":
        options["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        model=AutoModelForCausalLM.from_pretrained(item["model_id"],**options)
    else:model=AutoModelForImageTextToText.from_pretrained(item["model_id"],**options)
    require(model.config._commit_hash==item["revision"],"Loaded revision mismatch")
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    model.config.use_cache=not training
    return model


def encode(kind,processor,prompt,image=None):
    rendered=render_prompt(kind,processor,prompt)
    if kind=="llm":inputs=processor(rendered,return_tensors="pt",add_special_tokens=False)
    else:inputs=processor(text=[rendered],images=[image],return_tensors="pt",add_special_tokens=False)
    return inputs

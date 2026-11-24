"""Actual preference fine-tuning with TRL DPO + LoRA; smoke runs never count as coursework."""
import argparse
import importlib.metadata
import time
from pathlib import Path

from common import MODELS,ROOT,digest,read,require,save,sha,validate_data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=["llm","vlm"],required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();manifest=validate_data()
    output=ROOT/("Output/smoke" if args.smoke else "Output")/args.kind
    require(not (output/"training.json").exists(),"Training already exists; do not overwrite experiment evidence")
    if (output/"trainer").exists():
        require(not any((output/"trainer").iterdir()),"An incomplete trainer directory exists; inspect it and preserve or move it before a fresh run")
    import torch
    from datasets import Dataset,Image as DatasetImage,Sequence
    from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training
    from trl import DPOConfig,DPOTrainer
    from transformers import set_seed
    from models import load_model,processor_for,render_prompt
    set_seed(42)
    processor=processor_for(args.kind)
    def dataset(split):
        examples=read(ROOT/f"Data/{args.kind}_{split}.json")
        if args.smoke:examples=examples[:8 if split=="train" else 4]
        rows=[]
        for q in examples:
            row={"prompt":render_prompt(args.kind,processor,q["prompt"]),"chosen":q["chosen"],"rejected":q["rejected"]}
            if args.kind=="vlm":row["images"]=[str(ROOT/"Data"/q["image"])]
            else:
                # The fixed dataset is short enough to keep every prompt/completion token.
                length=len(processor(row["prompt"]+q["chosen"],add_special_tokens=False)["input_ids"])
                require(length<=1024,"Unexpected long training sequence; never silently truncate")
            rows.append(row)
        ds=Dataset.from_list(rows)
        if args.kind=="vlm":ds=ds.cast_column("images",Sequence(DatasetImage()))
        return ds
    train,dev=dataset("train"),dataset("dev")
    model=load_model(args.kind,training=True)
    if args.kind=="llm":model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False})
    targets=["q_proj","k_proj","v_proj","o_proj"] if args.kind=="llm" else r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj)$"
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=targets,bias="none",task_type=TaskType.CAUSAL_LM)
    model=get_peft_model(model,lora)
    trainable={name:p.detach().float().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
    require(bool(trainable),"No trainable adapter parameters")
    model.print_trainable_parameters()
    config=DPOConfig(output_dir=str(output/"trainer"),num_train_epochs=3,max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=4,
        learning_rate=5e-5 if args.kind=="llm" else 1e-4,beta=0.1,
        loss_type="sigmoid",bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
        max_length=None,max_prompt_length=None,max_completion_length=None,
        precompute_ref_log_probs=True,precompute_ref_batch_size=1,
        optim="adamw_torch",warmup_ratio=0.1,logging_steps=1,
        eval_strategy="epoch",save_strategy="epoch",save_total_limit=2,
        load_best_model_at_end=True,metric_for_best_model="eval_loss",greater_is_better=False,
        dataloader_num_workers=0,dataset_num_proc=None,remove_unused_columns=False,
        report_to="none",seed=42,data_seed=42,disable_dropout=True)
    trainer=DPOTrainer(model=model,ref_model=None,args=config,train_dataset=train,eval_dataset=dev,processing_class=processor)
    output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    outcome=trainer.train()
    elapsed=time.perf_counter()-start
    adapter=output/"adapter"
    trainer.save_model(str(adapter));processor.save_pretrained(str(output/"processor"))
    changed_squared=0.0
    for name,p in model.named_parameters():
        if name in trainable:changed_squared+=float(((p.detach().float().cpu()-trainable[name])**2).sum())
    require(changed_squared>0 and trainer.state.global_step>0,"No actual parameter update detected")
    meta={"schema_version":1,"kind":args.kind,"smoke":args.smoke,"complete":True,"algorithm":"DPO sigmoid preference optimization; LoRA adapters",
        "model":{k:v for k,v in MODELS[args.kind].items() if k!="cache"},"data_manifest_sha256":sha(ROOT/"Data/manifest.json"),
        "train_examples":len(train),"dev_examples":len(dev),"test_examples_used_for_training":0,
        "global_step":trainer.state.global_step,"epochs":trainer.state.epoch,"best_checkpoint":trainer.state.best_model_checkpoint,
        "selection":"Lowest dev DPO loss; no held-out generation scores used to select adapters",
        "trainable_parameters":sum(x.numel() for x in trainable.values()),"adapter_parameter_change_l2":changed_squared**0.5,
        "training_seconds":elapsed,"gpu":torch.cuda.get_device_name(0),"peak_allocated_bytes":torch.cuda.max_memory_allocated(0),
        "packages":{name:importlib.metadata.version(name) for name in ["torch","transformers","trl","peft","datasets","bitsandbytes"]},
        "trainer_args":config.to_dict(),"train_metrics":outcome.metrics,"log_history":trainer.state.log_history,
        "adapter_sha256":{p.name:sha(p) for p in adapter.iterdir() if p.is_file()}}
    save(output/"training.json",meta)
    print(f"Saved {'SMOKE ONLY' if args.smoke else 'formal'} training evidence: {output}",flush=True)


if __name__=="__main__":main()

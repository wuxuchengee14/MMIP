"""Actual pretrained encoders/scorer smoke check; not a homework test-set evaluation."""
import gc
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from common.utils import ROOT,configure,read_json,write_json
from quiz1.Code.prepare import week4_root
from quiz3.Code.features import ClipImages,load_encoder,patch_features


def main():
    config=read_json(ROOT/"config.rtx5090.json");device=configure(config,"cuda")
    from transformers import AutoModel,AutoTokenizer
    name=config["multimodal"]["text_model"]
    revision=config["multimodal"]["text_revision"]
    tokenizer=AutoTokenizer.from_pretrained(name,revision=revision)
    encoder=AutoModel.from_pretrained(name,revision=revision).to(device).eval().requires_grad_(False)
    tokens=tokenizer(["A small bird has a yellow chest.","This bird has long black wings."],padding=True,return_tensors="pt").to(device)
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):output=encoder(**tokens).last_hidden_state
    assert output.shape[0]==2 and output.shape[-1]==768 and torch.isfinite(output).all()
    result={"scope":"real pretrained encoders and evaluator; tiny smoke inputs only, not final accuracy",
            "distilbert":{"pass":True,"shape":list(output.shape),"revision":getattr(encoder.config,"_commit_hash",None)}}
    del encoder,output;gc.collect();torch.cuda.empty_cache()
    net,processor=load_encoder(config,device)
    folder=week4_root(config)/"quiz1/Data/flickr8k"
    rows=read_json(folder/"test.json")[:2]
    dataset=ClipImages(folder,rows,processor)
    pixels=torch.stack([dataset[i] for i in range(2)]).to(device)
    from PIL import Image
    with Image.open(folder/rows[0]["image"]) as im:official=processor(images=im.convert("RGB"),return_tensors="pt")["pixel_values"][0]
    delta=float((pixels[0].cpu()-official).abs().max());assert delta<1e-5,delta
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):features=patch_features(net,pixels,4)
    assert features.shape==(2,16,768) and torch.isfinite(features).all()
    result["clip"]={"pass":True,"shape":list(features.shape),"preprocessing_max_difference":delta,
                    "revision":net.mmip_revision}
    del net,features,pixels;gc.collect();torch.cuda.empty_cache()
    from bert_score import BERTScorer
    options=config["bertscore"]
    scorer=BERTScorer(model_type=options["model_type"],num_layers=options["num_layers"],device="cuda",
                     idf=False,rescale_with_baseline=False,use_fast_tokenizer=False)
    sentence="a dog is swimming in a pool ."
    p,r,f=scorer.score([sentence],[[sentence,"a dog swims in blue water ."]],batch_size=2)
    assert float(f[0])>.999 and torch.isfinite(f).all()
    result["bertscore"]={"pass":True,"identical_sentence_f1":float(f[0]),"hash":scorer.hash}
    write_json(ROOT/"analysis/pretrained_probe.json",result)
    print(result,flush=True)


if __name__=="__main__":main()

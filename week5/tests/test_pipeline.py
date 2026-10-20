"""CPU integration with tiny synthetic fixtures; no network or production data is used."""
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
import common.utils as utils
import common.reports as reports
import quiz1.Code.prepare as prepare
import quiz1.Code.train as retrieval
import quiz1.Code.models as retrieval_models
import quiz2.Code.train as fusion
import quiz2.Code.models as fusion_models
import quiz3.Code.train as caption
import quiz3.Code.compare as compare
from test_workflow import TinyBackbone


class FixtureScorer:
    hash="OFFLINE_FIXTURE_NOT_REAL_BERTSCORE"
    _model=SimpleNamespace(config=SimpleNamespace(_commit_hash=None))
    def __init__(self,**kwargs):pass
    def score(self,captions,references,**kwargs):
        if len(captions)!=len(references) or any(len(r)!=5 for r in references):raise ValueError("Bad references")
        return (torch.ones(len(captions)),)*3


class PipelineTests(unittest.TestCase):
    def test_five_models_reports_and_six_caption_comparisons(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp)/"week5";root.mkdir()
            old=Path(tmp)/"week4"
            for module in (utils,reports,prepare,retrieval,fusion,caption,compare):
                stack.enter_context(patch.object(module,"ROOT",root))
            config=utils.read_json(Path(__file__).resolve().parents[1]/"config.rtx5090.json")
            config.update(workers=0,cpu_threads=2,pin_memory=False)
            config["retrieval"].update(epochs=1,patience=1,classes_per_batch=2,images_per_class=2,embedding_dim=8,eval_batch_size=16,example_queries=2)
            config["multimodal"].update(epochs=1,patience=1,batch_size=8,fusion_dim=8)
            config["caption"].update(epochs=1,patience=1,batch_size=4,dim=16,layers=1,heads=2,grid_size=2,max_length=8,
                                     decode_batch_size=4,timing_samples=1,timing_repeats=1)
            folder=prepare.cub_root();image_root=folder/"raw/CUB_200_2011"
            records={};counter=0
            for split,per_class in (("train",4),("val",2),("test",4)):
                rows=[]
                for label in range(4):
                    for _ in range(per_class):
                        counter+=1;image=f"images/{counter}.jpg"
                        path=image_root/image;path.parent.mkdir(parents=True,exist_ok=True)
                        Image.new("RGB",(28,28),(label*50,counter*3,50)).save(path)
                        rows.append({"id":str(counter),"image":image,"label":label,"class_name":f"bird{label}",
                                     "captions":[f"a bird with colored feathers {j}" for j in range(10)]})
                records[split]=rows;utils.write_json(folder/f"{split}.json",rows)
            utils.write_json(folder/"classes.json",[f"bird{i}" for i in range(4)])
            manifest={"classes":4,"image_root":"raw/CUB_200_2011","fixture":True,"counts":{s:len(r) for s,r in records.items()}}
            utils.write_json(folder/"manifest.json",manifest)
            utils.write_json(folder/"retrieval_splits.json",{s:prepare.divide_query_gallery(records[s],42) for s in ("val","test")})
            with patch.object(retrieval_models,"ImageBackbone",TinyBackbone):
                result=retrieval.run(config,torch.device("cpu"))
            self.assertEqual(len(result),2)
            self.assertEqual(result[0]["queries"],8)
            for split,rows in records.items():
                utils.atomic_torch_save({"features":torch.randn(len(rows),10 if split=="train" else 1,6),
                                         "ids":[r["id"] for r in rows]},root/f"quiz2/Data/text_features/{split}.pt")
            text_meta={"identity":"fixture","dimension":6,"parameters":100}
            with patch.object(fusion_models,"ImageBackbone",TinyBackbone),patch.object(fusion,"prepare_text_features",return_value=text_meta):
                result=fusion.run(config,torch.device("cpu"))
            self.assertEqual(len(result),2)
            flickr=old/"quiz1/Data/flickr8k";vocab=["<pad>","<unk>","<bos>","<eos>","a","bird","on","the","branch","."]
            utils.write_json(flickr/"vocab.json",vocab)
            utils.write_json(old/"config.rtx5090.json",config)
            for split,size in (("train",4),("val",2),("test",4)):
                rows=[]
                for i in range(size):
                    image=f"{split}_{i}.jpg";path=flickr/image;path.parent.mkdir(parents=True,exist_ok=True)
                    Image.new("RGB",(28,28),(40,i*30,70)).save(path)
                    rows.append({"id":image,"image":image,"captions":["a bird on the branch ."]*5})
                utils.write_json(flickr/f"{split}.json",rows)
                utils.atomic_torch_save({"features":torch.randn(size,4,6).half(),"ids":[r["id"] for r in rows]},root/f"quiz3/Data/clip_features/{split}.pt")
                if split=="test":test_rows=rows
            feature_meta={"identity":"fixture","feature_shape":[4,6],"encoder_parameters":100,"extraction_seconds":0}
            with patch.object(caption,"prepare_features",return_value=feature_meta),patch.object(caption,"benchmark_encoder",return_value={"encoder_mean_ms":1}):
                result=caption.run(config,torch.device("cpu"))
            self.assertEqual(len(result),2)
            originals={}
            for kind in ("lstm","transformer"):
                for mode in ("greedy","beam"):
                    output=old/f"quiz4/Output/{kind}/{mode}"
                    utils.write_json(output/"predictions.json",[{"id":r["id"],"caption":"a bird on the branch .","references":r["captions"]} for r in test_rows])
                    utils.write_json(output/"timing.json",{"decoder_mean_ms":1,"estimated_pipeline_mean_ms":2})
                    originals[str(output/"predictions.json")]=utils.file_hash(output/"predictions.json")
            with patch("bert_score.BERTScorer",FixtureScorer):result=compare.run(config,torch.device("cpu"))
            self.assertEqual(len(result),6)
            with patch("bert_score.BERTScorer",side_effect=AssertionError("Must reuse existing score cache")):
                compare.run(config,torch.device("cpu"))
            for path,checksum in originals.items():self.assertEqual(utils.file_hash(path),checksum)
            reports.make_reports(config)
            self.assertFalse(reports.completion(config),"A four-image fixture must not count as full 1000-image evaluation")
            self.assertIn("CLIP",(root/"quiz3/Output/report.md").read_text(encoding="utf-8"))
            # Changing a reference must reject the old prediction, not silently recompute a different task.
            test_rows[0]["captions"][0]="changed reference"
            utils.write_json(flickr/"test.json",test_rows)
            with self.assertRaises(ValueError):compare.run(config,torch.device("cpu"))


if __name__=="__main__":unittest.main(verbosity=2)

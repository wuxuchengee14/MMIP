"""Offline tests use temporary synthetic fixtures, never production predictions or scores."""
from __future__ import annotations

import copy
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from common import utils
from common.data import extract,safe_path,download,download_ranges
from common.training import fit
from common.text import encode,tokenize
from quiz1.Code.models import supervised_contrastive,RetrievalModel
from quiz1.Code.prepare import parse_cub,divide_query_gallery
from quiz1.Code.retrieval import retrieval_metrics
from quiz1.Code.train import ClassBatchSampler
from quiz2.Code.models import FusionClassifier
from quiz2.Code.train import CaptionSampler
from quiz3.Code.models import CaptionModel,generate,decode
from quiz3.Code.bleu import corpus_bleu
from quiz3.Code.compare import aligned_predictions


class TinyBackbone(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__();self.dimension=8
        self.net=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,8))
    def forward(self,images):return self.net(images)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_supcon_has_finite_gradients_and_rejects_singletons(self):
        x=torch.randn(8,6,requires_grad=True);z=F.normalize(x,dim=1)
        loss=supervised_contrastive(z,torch.tensor([0,0,1,1,2,2,3,3]))
        loss.backward();self.assertTrue(torch.isfinite(x.grad).all())
        with self.assertRaises(ValueError):supervised_contrastive(z,torch.arange(8))
        grouped=torch.eye(4).repeat_interleave(2,dim=0)
        good=supervised_contrastive(grouped,torch.arange(4).repeat_interleave(2))
        bad=supervised_contrastive(grouped,torch.arange(4).repeat(2))
        self.assertLess(good,bad)

    def test_retrieval_metrics_multi_positive_and_self_exclusion(self):
        q=np.array([[1.,0.]])
        g=np.array([[1.,0.],[.99,.01],[.98,.02],[0.,1.]])
        metrics,ranking,_,_=retrieval_metrics(q,g,[0],[0,0,0,1],["self"],["self","a","b","c"],(1,2))
        self.assertNotEqual(int(ranking[0,0]),0)
        self.assertEqual(metrics["hit_at_1"],1)
        self.assertEqual(metrics["recall_at_1"],.5)
        self.assertEqual(metrics["map_at_2"],1)
        self.assertEqual(metrics["map"],1)
        with self.assertRaises(ValueError):retrieval_metrics(q,g,[9],[0,0,0,1],["q"],["s","a","b","c"])

    def test_pk_and_caption_samplers_are_epoch_resumable(self):
        labels=np.repeat(np.arange(4),6)
        sampler=ClassBatchSampler(labels,2,3,42)
        sampler.set_epoch(3);a=list(sampler)
        for batch in a:
            selected=[labels[i] for i,_ in batch]
            self.assertEqual(sorted(np.unique(selected,return_counts=True)[1]),[3,3])
            self.assertEqual(len({i for i,_ in batch}),6)
            self.assertTrue(all(epoch==3 for _,epoch in batch))
        sampler.set_epoch(3);self.assertEqual(a,list(sampler))
        captions=CaptionSampler(10,10,42);captions.set_epoch(4);first=list(captions)
        captions.set_epoch(4);self.assertEqual(first,list(captions))
        self.assertEqual({i for i,_,_ in first},set(range(10)))

    def test_archive_traversal_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ("../escape","..\\escape","C:/outside"):
                with self.assertRaises(ValueError):safe_path(root/"data",name)
            archive=root/"bad.tar"
            with tarfile.open(archive,"w") as tar:
                item=tarfile.TarInfo("link");item.type=tarfile.SYMTYPE;item.linkname="../escape";tar.addfile(item)
            with self.assertRaises(ValueError):extract(archive,root/"target")
            archive=root/"bad.zip"
            with zipfile.ZipFile(archive,"w") as z:z.writestr("../escape","bad")
            with self.assertRaises(ValueError):extract(archive,root/"target")
            self.assertFalse((root/"escape").exists())

    def test_partial_download_resumes_only_matching_range(self):
        class Response(io.BytesIO):
            status=206
            headers={"Content-Range":"bytes 3-5/6","Content-Length":"3"}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"file.bin";path.with_name("file.bin.part").write_bytes(b"abc")
            with patch("common.data.urllib.request.urlopen",return_value=Response(b"def")) as request,patch("common.data.enable_system_trust"):
                download("https://example.invalid/file",path)
            self.assertEqual(path.read_bytes(),b"abcdef")
            self.assertEqual(request.call_args.args[0].get_header("Range"),"bytes=3-")

    def test_parallel_ranges_preserve_existing_prefix_and_verify_bounds(self):
        raw=b"abcdefghijklmnopqrstuvwxyz"
        class Response(io.BytesIO):
            status=206
            def __init__(self,start,end):
                super().__init__(raw[start:end+1]);self.headers={"Content-Range":f"bytes {start}-{end}/{len(raw)}"}
        def respond(request,**kwargs):
            start,end=request.get_header("Range").removeprefix("bytes=").split("-")
            return Response(int(start),int(end))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"archive.bin";path.with_name(path.name+".part").write_bytes(raw[:5])
            with patch("common.data.urllib.request.urlopen",side_effect=respond):
                self.assertTrue(download_ranges("https://example.invalid",path,4))
            self.assertEqual(path.read_bytes(),raw)
            self.assertFalse(path.with_name(path.name+".part").exists())

    def test_caption_causality_generation_and_bleu(self):
        model=CaptionModel("transformer",12,dim=16,layers=1,heads=2,dropout=0,max_length=8,grid_size=2,feature_dim=6).eval()
        features=torch.randn(2,4,6);tokens=torch.tensor([[2,4,5,3],[2,5,6,3]])
        a=model(features,tokens);changed=tokens.clone();changed[:,-1]=9
        b=model(features,changed);torch.testing.assert_close(a[:,:-1],b[:,:-1])
        a.square().mean().backward()
        for width in (1,3):
            result=generate(model,features,8,width)
            self.assertEqual(result.shape[0],2);self.assertFalse((result[:,1:]==0).any());self.assertFalse((result[:,1:]==2).any())
        self.assertAlmostEqual(corpus_bleu([["a cat sits on the mat"]],["a cat sits on the mat"])["bleu4"],1)
        self.assertEqual(corpus_bleu([["a cat"]],[""])["bleu4"],0)
        self.assertEqual(encode("unknown",{"a":4},4,True),[2,1,3])

    def test_both_fusion_modalities_affect_logits(self):
        import common.vision
        with patch("quiz2.Code.models.ImageBackbone",TinyBackbone):
            model=FusionClassifier("fixture",4,text_dim=6,fusion_dim=8,dropout=0).eval()
            images=torch.randn(3,3,8,8);text=torch.randn(3,6)
            output=model(images,text)
            self.assertFalse(torch.allclose(output,model(images+1,text)))
            self.assertFalse(torch.allclose(output,model(images,text+torch.arange(6))))
            output.square().mean().backward()
            self.assertIsNotNone(model.text_projection[0].weight.grad)
            self.assertIsNotNone(model.image_encoder.net[-1].weight.grad)

    def test_preprocessing_keeps_ten_descriptions_with_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=root/"raw";captions=root/"text";images.mkdir();captions.mkdir()
            names=[];labels=[];flags=[]
            for i in range(24):
                sid=str(i+1);label=i//12+1;name=f"{label:03d}.Bird/image{i}.jpg"
                target=images/"images"/name;target.parent.mkdir(parents=True,exist_ok=True)
                Image.new("RGB",(8,8),(i*9,20,40)).save(target)
                text=captions/Path(name).with_suffix(".txt");text.parent.mkdir(parents=True,exist_ok=True)
                text.write_text("\n".join(f"bird visual description {j}" for j in range(10)),encoding="utf-8")
                names.append(f"{sid} {name}");labels.append(f"{sid} {label}");flags.append(f"{sid} {int(i%12<8)}")
            for filename,values in (("images.txt",names),("image_class_labels.txt",labels),("train_test_split.txt",flags),("classes.txt",["1 BirdA","2 BirdB"])):
                (images/filename).write_text("\n".join(values))
            config={"expected_images":24,"expected_classes":2,"captions_per_image":10,"validation_fraction":.25}
            splits,classes=parse_cub(images,captions,config,42)
            self.assertEqual(len(splits["test"]),8)
            self.assertEqual(len(classes),2)
            seen=set()
            for rows in splits.values():
                self.assertFalse(seen & {r["id"] for r in rows});seen.update(r["id"] for r in rows)
                self.assertTrue(all(len(r["captions"])==10 for r in rows))
            split=divide_query_gallery(splits["test"],42)
            self.assertFalse(set(split["query"])&set(split["gallery"]))
            next(captions.rglob("*.txt")).unlink()
            with self.assertRaises(FileNotFoundError):parse_cub(images,captions,config,42)

    def test_prediction_alignment_rejects_different_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"predictions.json"
            records=[{"id":"a","captions":["a cat"]*5}]
            utils.write_json(path,[{"id":"a","caption":"a cat","references":["a dog"]*5}])
            with self.assertRaises(ValueError):aligned_predictions(path,records)
            utils.write_json(path,[{"id":"a","caption":"a cat","references":["a cat"]*5}])
            self.assertEqual(aligned_predictions(path,records),["a cat"])

    def test_epoch_resume_matches_uninterrupted(self):
        data=TensorDataset(torch.arange(16).float().reshape(8,2)/16,torch.arange(8)%2)
        settings={"epochs":3,"patience":5,"lr":.01,"weight_decay":0,"clip":1}
        def step(model,batch,device):
            x,y=batch;output=model(x);return F.cross_entropy(output,y),len(y),int(output.argmax(1).eq(y).sum())
        with tempfile.TemporaryDirectory() as tmp:
            def run(name,callback):
                utils.seed_all(42);model=nn.Sequential(nn.Linear(2,8),nn.Dropout(.2),nn.Linear(8,2))
                fit(model,DataLoader(data,batch_size=4,shuffle=True),DataLoader(data,batch_size=4),callback,settings,
                    Path(tmp)/name,torch.device("cpu"),{"seed":42,"fixture":True})
                return model
            baseline=run("base",step);calls=0
            def interrupted(*args):
                nonlocal calls
                calls+=1
                if calls==5:raise KeyboardInterrupt()
                return step(*args)
            with self.assertRaises(KeyboardInterrupt):run("resume",interrupted)
            resumed=run("resume",step)
            for a,b in zip(baseline.parameters(),resumed.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_notebooks_do_not_autotrain(self):
        import nbformat
        for n in range(1,4):
            note=nbformat.read(Path(__file__).resolve().parents[1]/f"quiz{n}/Code/quiz{n}.ipynb",as_version=4)
            nbformat.validate(note)
            self.assertIn("RUN_STAGE = False","\n".join(c.source for c in note.cells if c.cell_type=="code"))


if __name__=="__main__":unittest.main(verbosity=2)

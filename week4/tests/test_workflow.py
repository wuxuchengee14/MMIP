"""Offline tests: no real dataset downloads, pretrained downloads or API calls."""
from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import common.utils as utils
from common.training import fit
from common.metrics import classification_metrics
from quiz1.Code.prepare import encode, safe_destination, vocabulary
from quiz2.Code.models import SentimentModel
from quiz4.Code.bleu import corpus_bleu
from quiz4.Code.models import CaptionModel, decode, generate


def tiny_classifier():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 11))
    return model, model[-1]


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_bleu_reference_clipping_and_brevity(self):
        perfect = corpus_bleu([["a cat sits on the mat", "the cat is on a mat"]],
                              ["a cat sits on the mat"])
        self.assertAlmostEqual(perfect["bleu4"], 1.0)
        clipped = corpus_bleu([["the cat", "a cat"]], ["the the the the"])
        self.assertAlmostEqual(clipped["bleu1"], 0.25)
        short = corpus_bleu([["a cat sits on a mat"]], ["a cat"])
        self.assertAlmostEqual(short["bleu1"], np.exp(-2))
        self.assertEqual(corpus_bleu([["a cat"]], [""])["bleu4"], 0.0)

    def test_training_only_vocab_and_caption_boundaries(self):
        vocab = vocabulary(["training only token"], 20)
        words = {w: i for i, w in enumerate(vocab)}
        self.assertEqual(encode("unseen", words, 4), [1])
        ids = encode("training only token token token", words, 4, caption=True)
        self.assertEqual((ids[0], ids[-1], len(ids)), (2, 3, 4))
        with self.assertRaises(ValueError):
            safe_destination(Path(tempfile.gettempdir()) / "archive", "../../escape")

    def test_multiclass_metrics_use_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = np.array([0, 1, 2, 0, 1, 2])
            probs = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]] * 2)
            metrics = classification_metrics(labels, probs, ["a", "b", "c"], Path(tmp))
            self.assertEqual(metrics["top1"], 1.0)
            self.assertEqual(metrics["macro_auc"], 1.0)
            self.assertEqual(metrics["macro_f1"], 1.0)

    def test_sentiment_padding_does_not_change_prediction(self):
        for kind in ("rnn", "lstm"):
            model = SentimentModel(kind, 20, 8, 12, dropout=0).eval()
            first = torch.tensor([[4, 5, 6]])
            padded = torch.tensor([[4, 5, 6, 0, 0, 0]])
            with torch.no_grad():
                a = model(first, torch.tensor([3]))
                b = model(padded, torch.tensor([3]))
            torch.testing.assert_close(a, b)

    def test_caption_causality_and_both_backwards(self):
        for kind in ("lstm", "transformer"):
            model = CaptionModel(kind, 12, dim=16, layers=1, heads=2, dropout=0,
                                 max_length=8, grid_size=2, feature_dim=6)
            features = torch.randn(2, 4, 6)
            tokens = torch.tensor([[2, 4, 5, 3], [2, 5, 6, 3]])
            model.eval()
            a = model(features, tokens)
            changed = tokens.clone()
            changed[:, -1] = 8
            b = model(features, changed)
            torch.testing.assert_close(a[:, :-1], b[:, :-1])
            model.train()
            loss = model(features, tokens).square().mean()
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            model.eval()
            for width in (1, 3):
                result = generate(model, features, 8, width)
                self.assertEqual(result.shape[0], 2)
                self.assertTrue((result[:, 0] == 2).all())
                self.assertFalse((result[:, 1:] == 0).any())
                self.assertFalse((result[:, 1:] == 2).any())

    def test_epoch_resume_matches_uninterrupted_and_rejects_changed_config(self):
        settings = {"epochs": 3, "patience": 4, "lr": 0.01, "weight_decay": 0, "clip": 1}
        identity = {"seed": 42, "model": "fixture"}
        data = TensorDataset(torch.arange(16).float().view(8, 2)/16, torch.arange(8) % 2)
        def step(model, batch, device):
            x, y = batch
            logits = model(x)
            return F.cross_entropy(logits, y), len(y), int((logits.argmax(1) == y).sum())
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            utils.seed_all(42)
            initial = nn.Sequential(nn.Linear(2, 8), nn.Dropout(0.2), nn.Linear(8, 2))
            saved = copy.deepcopy(initial.state_dict())
            def train(name, callback):
                utils.seed_all(42)
                model = nn.Sequential(nn.Linear(2, 8), nn.Dropout(0.2), nn.Linear(8, 2))
                model.load_state_dict(saved)
                fit(model, DataLoader(data, batch_size=4, shuffle=True), DataLoader(data, batch_size=4),
                    callback, settings, directory/name, torch.device("cpu"), identity)
                return model
            baseline = train("baseline", step)
            calls = 0
            def interrupted(model, batch, device):
                nonlocal calls
                calls += 1
                if calls == 5:
                    raise KeyboardInterrupt()
                return step(model, batch, device)
            with self.assertRaises(KeyboardInterrupt):
                train("resume", interrupted)
            resumed = train("resume", step)
            for a, b in zip(baseline.parameters(), resumed.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            a = torch.load(directory/"baseline"/"last.pt", weights_only=False)["model"]
            b = torch.load(directory/"resume"/"last.pt", weights_only=False)["model"]
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            with self.assertRaises(RuntimeError):
                fit(resumed, [], [], step, dict(settings, lr=0.02), directory/"resume",
                    torch.device("cpu"), identity)

    def test_synthetic_complete_pipeline_and_gemini_cache(self):
        import run_all
        import quiz2.Code.train as text
        import quiz3.Code.train as vision
        import quiz4.Code.features as features
        import quiz4.Code.train as caption
        import quiz4.Code.gemini as gemini
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            for module in (utils, run_all, text, vision, features, caption, gemini):
                stack.enter_context(patch.object(module, "ROOT", root))
            config = utils.read_json(Path(__file__).resolve().parents[1]/"config.rtx5090.json")
            config.update(workers=0, cpu_threads=2)
            config["text"].update(epochs=1, patience=1, batch_size=4, hidden_dim=12, embedding_dim=8)
            config["vision"].update(epochs=1, patience=1, batch_size=11, warmup_epochs=1)
            config["caption"].update(epochs=1, patience=1, batch_size=4, dim=16, heads=2,
                                     layers=1, max_length=8, timing_samples=1, timing_repeats=1,
                                     gemini_samples=2, gemini_interval_seconds=0)
            folder = root/"quiz1"/"Data"/"imdb"
            folder.mkdir(parents=True)
            vocab = ["<pad>", "<unk>", "<bos>", "<eos>", "good", "bad", "movie"]
            utils.write_json(folder/"vocab.json", vocab)
            utils.write_json(folder/"manifest.json", {"fixture": True})
            for split in ("train", "val", "test"):
                ids = [f"{split}/{'pos' if i%2 else 'neg'}/{i}.txt" for i in range(8)]
                tokens = torch.tensor([[4 if i%2 else 5, 6, 0, 0] for i in range(8)])
                torch.save({"tokens": tokens, "lengths": torch.full((8,), 2),
                            "labels": torch.arange(8)%2, "ids": ids}, folder/f"{split}.pt")
                for sid in ids:
                    p = folder/"raw"/"aclImdb"/sid
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("good movie" if "/pos/" in sid else "bad movie", encoding="utf-8")
            text.run(config, torch.device("cpu"))
            folder = root/"quiz1"/"Data"/"organamnist"
            folder.mkdir(parents=True)
            utils.write_json(folder/"manifest.json", {"classes": [f"class{i}" for i in range(11)]})
            arrays = {f"{s}_{k}": (np.random.default_rng(42).integers(0, 255, (11, 28, 28), dtype=np.uint8)
                                     if k == "images" else np.arange(11).reshape(-1, 1))
                      for s in ("train", "val", "test") for k in ("images", "labels")}
            np.savez(folder/"organamnist.npz", **arrays)
            with patch.object(vision, "build_model", side_effect=lambda _: tiny_classifier()):
                vision.run(config, torch.device("cpu"))
            folder = root/"quiz1"/"Data"/"flickr8k"
            folder.mkdir(parents=True)
            utils.write_json(folder/"vocab.json", vocab + ["a", "cat", "sits", "here"])
            cache = root/"quiz4"/"Data"/"features"
            cache.mkdir(parents=True)
            feature_info = {"identity": "fixture", "extraction_seconds": 0.0, "encoder_parameters": 100}
            utils.write_json(cache/"complete.json", feature_info)
            for split in ("train", "val", "test"):
                rows = []
                for i in range(2):
                    name = f"{split}{i}.jpg"
                    Image.new("RGB", (20, 20), (i*100, 80, 40)).save(folder/name)
                    rows.append({"id": name, "image": name, "captions": ["a cat sits here"]*5})
                utils.write_json(folder/f"{split}.json", rows)
                torch.save({"features": torch.randn(2, 16, 2048).half(),
                            "ids": [r["id"] for r in rows]}, cache/f"{split}.pt")
            with patch.object(caption, "prepare_features", return_value=feature_info), \
                 patch.object(caption, "benchmark_encoder", return_value={"encoder_mean_ms": 1.0}):
                caption.run(config, torch.device("cpu"))
            calls = []
            def fake_request(key, model, image, candidate):
                calls.append(image)
                return ({"verdict": "partial", "score": 3, "subject_correct": True,
                         "action_correct": False, "scene_correct": True, "reason": "Mock only"},
                        {"fixture": True})
            stack.enter_context(patch.dict(os.environ, {"GEMINI_API_KEY": "FAKE_NOT_A_REAL_KEY"}))
            with patch.object(gemini, "request_judgment", side_effect=fake_request):
                self.assertTrue(gemini.run(config))
                self.assertTrue(gemini.run(config))
            self.assertEqual(len(calls), 4, "Second run must reuse successful Gemini cache")
            caption.make_report()
            utils.write_json(root/"quiz1"/"Output"/"datasets.json", {"fixture": True})
            (root/"quiz1"/"Output"/"report.md").write_text("fixture", encoding="utf-8")
            self.assertTrue(run_all.completion(config))
            with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
                self.assertFalse(gemini.run(config))
            self.assertFalse(run_all.completion(config))
            caption.make_report()
            self.assertIn("未完成", (root/"quiz4"/"Output"/"report.md").read_text(encoding="utf-8"))

    def test_notebooks_do_not_autotrain(self):
        import nbformat
        root = Path(__file__).resolve().parents[1]
        for n in range(1, 5):
            notebook = nbformat.read(root/f"quiz{n}"/"Code"/f"quiz{n}.ipynb", as_version=4)
            nbformat.validate(notebook)
            source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
            self.assertIn("RUN_STAGE = False", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)

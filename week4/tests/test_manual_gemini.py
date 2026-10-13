"""Offline manual import checks; synthetic answers stay in temporary directories."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from common.utils import read_json, write_json
from quiz4.Code import manual_gemini as manual


class ManualTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.root_patch = patch.object(manual, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        folder = self.root / "quiz1/Data/flickr8k"
        folder.mkdir(parents=True)
        ids = [f"image{i}.jpg" for i in range(6)]
        for i, sid in enumerate(ids):
            Image.new("RGB", (8, 8), (i * 30, 20, 40)).save(folder / sid)
        write_json(self.root / "quiz4/Output/gemini/sample_ids.json", ids)
        for kind in ("lstm", "transformer"):
            rows = [{"id": sid, "image": sid, "caption": kind + " a cat"} for sid in ids]
            write_json(self.root / f"quiz4/Output/{kind}/greedy/predictions.json", rows)
        manual.prepare()
        self.output = manual.directory()
        self.manifest = read_json(self.output / "manifest.json")
        write_json(self.output / "session.json", {"source": "manual_upload_gemini_web",
            "evaluator_model_label": "Offline fixture", "operator": "Fixture", "review_started_on": "2026-10-13"})

    def payload(self, batch):
        return {"batch_id": batch["batch_id"], "judgments": [
            {"image_id": e["image_id"], "candidate_id": c, "verdict": "partial", "score": 3,
             "subject_correct": True, "action_correct": False, "scene_correct": True,
             "reason": "Synthetic test only"} for e in batch["entries"] for c in ("A", "B")]}

    def save_batch(self, batch, payload=None):
        path = self.output / "responses" / (batch["batch_id"] + ".json")
        write_json(path, self.payload(batch) if payload is None else payload)
        return path

    def test_prepare_balances_candidates_and_zip_has_no_answers(self):
        entries = [e for b in self.manifest["batches"] for e in b["entries"]]
        self.assertEqual(sum(e["candidates"]["A"]["model"] == "lstm" for e in entries), 3)
        self.assertEqual(len(list((self.output / "uploads").rglob("*.jpg"))), 6)
        with ZipFile(self.output / "upload_batches.zip") as archive:
            self.assertFalse(any("responses/" in n or "manifest.json" in n for n in archive.namelist()))
        self.assertFalse(read_json(self.output / "summary.json")["complete"])
        manual.prepare()  # Idempotent; preserve session metadata.
        self.assertEqual(read_json(self.output / "session.json")["operator"], "Fixture")

    def test_missing_and_duplicate_candidates_rejected(self):
        batch = self.manifest["batches"][0]
        for duplicate in (False, True):
            payload = self.payload(batch)
            if duplicate:
                payload["judgments"].append(copy.deepcopy(payload["judgments"][0]))
            else:
                payload["judgments"].pop()
            with self.assertRaises(ValueError):
                manual.parse_response(self.save_batch(batch, payload), batch)

    def test_bad_ids_scores_flags_and_verdicts_rejected(self):
        batch = self.manifest["batches"][0]
        for field, value in (("image_id", "unknown.jpg"), ("candidate_id", "C"), ("score", 6),
                             ("score", True), ("subject_correct", "true"), ("verdict", "match")):
            payload = self.payload(batch)
            payload["judgments"][0][field] = value
            with self.assertRaises(ValueError):
                manual.parse_response(self.save_batch(batch, payload), batch)

    def test_fenced_json_accepted(self):
        batch = self.manifest["batches"][0]
        path = self.save_batch(batch)
        path.write_text("```json\n" + path.read_text() + "\n```", encoding="utf-8")
        self.assertEqual(len(manual.parse_response(path, batch)), 10)

    def test_partial_then_full_import_and_completion_source(self):
        import run_all
        batches = self.manifest["batches"]
        self.save_batch(batches[0])
        self.assertFalse(manual.import_all())
        self.assertEqual(read_json(self.output / "summary.json")["successful"], 10)
        self.save_batch(batches[1])
        self.assertTrue(manual.import_all())
        summary = read_json(self.output / "summary.json")
        self.assertEqual(summary["successful"], 12)
        self.assertEqual([r["successful"] for r in summary["comparison"]], [6, 6])
        with patch.object(run_all, "ROOT", self.root):
            run_all.completion({"caption": {"gemini_samples": 6}})
            check = read_json(self.root / "completion.json")["checks"][-1]
            self.assertTrue(check["complete"])
            self.assertEqual(check["source"], "manual_upload_gemini_web")
            run_all.completion({"caption": {"gemini_samples": 100}})
            self.assertFalse(read_json(self.root / "completion.json")["checks"][-1]["complete"])

    def test_changed_evaluator_session_cannot_mix_results(self):
        for batch in self.manifest["batches"]:
            self.save_batch(batch)
        self.assertTrue(manual.import_all())
        session = read_json(self.output / "session.json")
        session["evaluator_model_label"] = "A different evaluator"
        write_json(self.output / "session.json", session)
        self.assertFalse(manual.import_all())
        self.assertEqual(read_json(self.output / "summary.json")["successful"], 0)

    def test_changed_caption_cannot_reuse_review(self):
        path = self.root / "quiz4/Output/lstm/greedy/predictions.json"
        rows = read_json(path)
        rows[0]["caption"] = "changed"
        write_json(path, rows)
        with self.assertRaises(ValueError):
            manual.import_all()
        with self.assertRaises(ValueError):
            manual.prepare()

    def test_duplicate_response_files_cannot_complete(self):
        for batch in self.manifest["batches"]:
            path = self.save_batch(batch)
            path.with_suffix(".txt").write_text(path.read_text(), encoding="utf-8")
        self.assertFalse(manual.import_all())
        self.assertEqual(read_json(self.output / "summary.json")["successful"], 0)

    def test_case_study_is_separate_and_accepts_only_twenty_judgments(self):
        import run_all
        # Extend the six-image fixture to ten, without using production images/answers.
        folder = self.root / "quiz1/Data/flickr8k"
        for i in range(6, 10):
            Image.new("RGB", (8, 8), (i*20, 20, 40)).save(folder / f"image{i}.jpg")
        for kind in ("lstm", "transformer"):
            path = self.root / f"quiz4/Output/{kind}/greedy/predictions.json"
            rows = read_json(path)
            rows.extend({"id": f"image{i}.jpg", "image": f"image{i}.jpg", "caption": kind + " a cat"}
                        for i in range(6, 10))
            write_json(path, rows)
        selected = [(f"image{i}.jpg", "Offline test selection") for i in range(10)]
        with patch.object(manual, "CASE_STUDY", True), patch.object(manual, "CASE_STUDY_SELECTION", selected):
            manual.prepare()
            self.output = manual.directory()
            self.assertEqual(self.output.name, "gemini_cases10")
            write_json(self.output / "session.json", {"source": manual.review_source(),
                "evaluator_model_label": "Offline fixture", "operator": "Fixture", "review_started_on": "2026-10-13"})
            manifest = read_json(self.output / "manifest.json")
            self.assertEqual(len(manifest["batches"]), 2)
            self.save_batch(manifest["batches"][0])
            self.assertFalse(manual.import_all())
            with patch.object(run_all, "ROOT", self.root):
                run_all.completion({"caption": {"gemini_samples": 100}})
                self.assertFalse(read_json(self.root / "completion.json")["checks"][-1]["complete"])
            self.save_batch(manifest["batches"][1])
            self.assertTrue(manual.import_all())
            summary = read_json(self.output / "summary.json")
            self.assertEqual(summary["successful"], 20)
            self.assertEqual(summary["selection_method"], "purposive_case_study")
            with patch.object(run_all, "ROOT", self.root):
                run_all.completion({"caption": {"gemini_samples": 100}})
                check = read_json(self.root / "completion.json")["checks"][-1]
                self.assertTrue(check["complete"])
                self.assertEqual(check["source"], "manual_upload_gemini_web_case_study")
                self.assertIn("案例展示", (self.root / "REPORT.md").read_text(encoding="utf-8"))
        self.assertEqual(read_json(self.root / "quiz4/Output/gemini_manual/summary.json")["successful"], 0)
        self.assertEqual(len(read_json(self.root / "quiz4/Output/gemini/sample_ids.json")), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

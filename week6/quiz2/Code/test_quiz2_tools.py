"""Offline tests. All synthetic model/review outputs stay in temporary directories."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import quiz2_tasks as tasks
from common import GENERATION, MODELS, ROOT, digest, load_benchmark, read, request_identity, save, sha


class ScoringTests(unittest.TestCase):
    def test_objective_metrics_and_invalid_boxes(self):
        questions = load_benchmark(ROOT/"Data/benchmark.json")
        self.assertEqual(tasks.objective(questions[0],"Dogs.")["score"],1)
        self.assertEqual(tasks.objective(questions[0],"There are dogs.")["score"],0)
        self.assertEqual(tasks.objective(questions[10],"LAB 204")["cer"],0)
        self.assertAlmostEqual(tasks.objective(questions[10],"LAB 205")["cer"],1/7)
        self.assertEqual(tasks.objective(questions[16],"a photo")["score"],None)
        self.assertEqual(tasks.bbox_iou([0,0,100,100],[0,0,100,100]),1)
        self.assertAlmostEqual(tasks.bbox_iou([50,0,150,100],[0,0,100,100]),1/3)
        for box in [None,{},[1,2,3], [1,2,0,4],[-1,2,3,4],[False,2,3,4],[0,0,1001,100],[0,0,float("nan"),1]]:
            self.assertEqual(tasks.bbox_iou(box,[0,0,100,100]),0)
        self.assertEqual(tasks.objective(questions[14],"```json\n[200,200,600,600]\n```")["score"],1)
        self.assertEqual(tasks.objective(questions[14],'[200,200,"600",600]')["score"],0)

    def test_strict_review_json(self):
        for text in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', 'prefix {"a":1}']:
            with self.assertRaises(ValueError):
                tasks.clean_json(text)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runs = self.root/"Output/full"
        self.judge = self.root/"Output/Commercial_manual"
        self.benchmark = ROOT/"Data/benchmark.json"
        self.questions = load_benchmark(self.benchmark)
        self.patch = patch.object(tasks,"ROOT",self.root)
        self.patch.start()
        save(self.root/"Data/models.lock.json",{"models":{key:{"model_id":model,"revision":"a"*40} for key,model in MODELS.items()}})
        for key,model in MODELS.items():
            rows = []
            for q in self.questions:
                answer = q["reference"]
                if isinstance(answer,list):
                    import json
                    answer = json.dumps(answer)
                rows.append({**request_identity(q),"answer":answer,"generation_seconds":0.1,
                             "preprocess_and_generation_seconds":0.2,"output_tokens":10,"hit_token_limit":False})
            save(self.runs/(key+".json"),{"schema_version":1,"model_key":key,"model_id":model,
                 "revision":"a"*40,"precision":"bf16","attention":"sdpa","batch_size":1,"seed":42,
                 "warmup_runs":1,"generation_config":GENERATION,"repetition_penalty":1.0,"length_penalty":1.0,
                 "benchmark_digest":digest(self.questions),"request_digest":digest([request_identity(q) for q in self.questions]),
                 "complete":True,"parameter_count":100,"gpu":"fixture GPU","packages":{"fixture":"1"},
                 "native_token_ids":{"bos":1,"eos":2,"pad":0},"results":rows})

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def populate_reviews(self):
        tasks.export_judge(self.benchmark,self.runs,self.judge)
        for q in self.questions:
            save(self.judge/"responses"/(q["id"]+".json"),{"question_id":q["id"],"image_seen":True,
                 "image_observations":"Unit-test fixture only; not a real commercial evaluation.",
                 "ratings":{"A":{"score":1,"reason":"test fixture"},"B":{"score":0.5,"reason":"test fixture"}}})
        tasks.import_judge(self.benchmark,self.runs,self.judge,"Unit test fixture, NOT a commercial model","2026-11-03")

    def test_full_workflow_and_stale_reviews(self):
        questions,runs = tasks.validated(self.benchmark,self.runs)
        scores = tasks.quantitative(questions,runs)
        self.assertEqual(scores["qwen"]["objective_questions"],18)
        self.assertEqual(scores["qwen"]["objective_accuracy_percent"],100)
        self.populate_reviews()
        tasks.report(self.benchmark,self.runs,self.judge)
        completion = read(self.runs/"completion.json")
        self.assertTrue(completion["all_complete"])
        self.assertEqual(completion["covered_points"],20)
        self.assertTrue((self.runs/"report.md").exists())
        self.assertTrue((self.runs/"comparison.csv").exists())
        # A changed answer invalidates all reviews, even if the question identity stays the same.
        run = read(self.runs/"qwen.json")
        run["results"][0]["answer"] = "cats"
        save(self.runs/"qwen.json",run)
        with self.assertRaisesRegex(ValueError,"answers changed"):
            tasks.import_judge(self.benchmark,self.runs,self.judge)

    def test_unfilled_review_and_unseen_image_rejected(self):
        tasks.export_judge(self.benchmark,self.runs,self.judge)
        with self.assertRaisesRegex(ValueError,"evaluator_model_label"):
            tasks.import_judge(self.benchmark,self.runs,self.judge)
        with self.assertRaisesRegex(ValueError,"viewing the image"):
            tasks.import_judge(self.benchmark,self.runs,self.judge,"Fixture","2026-11-03")
        self.assertFalse((self.runs/"completion.json").exists())
        self.assertFalse((self.runs/"commercial_scores.json").exists())

    def test_changed_prompt_and_settings_rejected(self):
        path = self.runs/"smol.json"
        original = read(path)
        mutations = [lambda x:x["results"][0].update(prompt="Another task"),
                     lambda x:x["generation_config"].update(do_sample=True),
                     lambda x:x.update(complete=False),
                     lambda x:x["results"][0].update(generation_seconds=float("nan"))]
        for mutation in mutations:
            changed = copy.deepcopy(original)
            mutation(changed)
            # JSON NaN isn't valid, so the common atomic writer rejects it before validation.
            try:
                save(path,changed)
            except ValueError:
                continue
            with self.assertRaises(ValueError):
                tasks.validated(self.benchmark,self.runs)
        save(path,original)

    def test_review_image_tamper_rejected(self):
        self.populate_reviews()
        path = self.judge/"cases/vision_01/image.png"
        path.write_bytes(path.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError,"image changed"):
            tasks.import_judge(self.benchmark,self.runs,self.judge)

    def test_benchmark_image_tamper_rejected(self):
        import shutil
        data = self.root/"benchmark_fixture"
        shutil.copytree(self.benchmark.parent,data)
        path = data/self.questions[0]["image"]
        path.write_bytes(path.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError,"Image changed"):
            load_benchmark(data/"benchmark.json")


if __name__ == "__main__":
    unittest.main()

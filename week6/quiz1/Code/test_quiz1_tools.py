"""Offline fixtures only; no real model, GPU, network or commercial review."""
import copy
import json
import tempfile
import time
import unittest
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gpu_memory as memory
import quiz1_tasks as tasks


class ToolTests(unittest.TestCase):
    def test_runner_saves_profile_with_stubbed_model_and_gpu(self):
        import qwen_local
        calls=[];closed=[]
        class Tokens:
            def __len__(self):return 2
            def cpu(self):return self
            def tolist(self):return [100,3]
        class Sequence:
            def __getitem__(self,key):return Tokens()
        class Batch(dict):
            def __init__(self):super().__init__(input_ids=SimpleNamespace(shape=(1,4)))
            def to(self,device):return self
        class Tokenizer:
            eos_token_id=3
            def apply_chat_template(self,*args,**kwargs):return Batch()
            def decode(self,*args,**kwargs):return "OFFLINE_FIXTURE_ANSWER"
        class Model:
            config=SimpleNamespace(_commit_hash="TEST_ONLY")
            generation_config=SimpleNamespace(eos_token_id=3)
            device="cuda:0"
            is_loaded_in_8bit=False
            def __init__(self,**options):self.is_loaded_in_4bit=bool(options.get('quantization_config'))
            def eval(self):return self
            def generate(self,**kwargs):calls.append(kwargs);return Sequence()
        class Monitor:
            name="FIXTURE_GPU";uuid="TEST_ONLY"
            def __init__(self,index):pass
            def read(self):return {"device_used_bytes":300,"torch_allocated_bytes":150,"torch_reserved_bytes":200}
            def close(self):closed.append(True)
        cuda=SimpleNamespace(is_available=lambda:True,synchronize=lambda:None,
            get_device_name=lambda index:"FIXTURE_GPU",reset_peak_memory_stats=lambda index:None,
            max_memory_allocated=lambda index:180)
        fake_torch=SimpleNamespace(cuda=cuda,bfloat16="bf16",inference_mode=nullcontext,
                                   __version__="TEST_ONLY",version=SimpleNamespace(cuda="TEST_ONLY"))
        fake_transformers=SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a,**kw:Model(**kw)),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:Tokenizer()),
            BitsAndBytesConfig=lambda **kw:kw,set_seed=lambda seed:None)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);questions=tasks.read(tasks.ROOT/'Data/benchmark.json')[:2]
            tasks.save(folder/'questions.json',questions)
            argv=['qwen_local.py','--precision','nf4','--questions',str(folder/'questions.json'),
                  '--output',str(folder/'output.json'),'--profile']
            with patch.dict(sys.modules,{'torch':fake_torch,'transformers':fake_transformers,
                       'truststore':SimpleNamespace(inject_into_ssl=lambda:None)}), \
                 patch.object(sys,'argv',argv),patch.object(memory,'NvidiaMemory',Monitor):
                qwen_local.main()
            result=tasks.read(folder/'output.json')
            self.assertEqual(len(calls),3)  # fixed warmup plus two actual fixture questions
            self.assertEqual(len(result['results']),2)
            self.assertTrue(result['loaded_in_4bit'])
            self.assertTrue(closed)
            summary=tasks.validate_memory(result)
            self.assertAlmostEqual(summary['device_used_bytes_mean'],300)

    def test_time_weighted_average_and_window(self):
        samples=[{"seconds":0,"bytes":100},{"seconds":1,"bytes":200},{"seconds":3,"bytes":400}]
        result=memory.summarize(samples)
        self.assertEqual(result["bytes_mean"],250)
        self.assertEqual(result["bytes_peak"],400)
        with self.assertRaises(ValueError):memory.summarize([samples[0]])
        window=memory.MemoryWindow(lambda:{"bytes":123},interval=.003).start()
        time.sleep(.015)
        measured=window.stop()
        self.assertGreaterEqual(len(measured["samples"]),2)
        self.assertAlmostEqual(measured["summary"]["bytes_mean"],123)

    def test_objective_rules(self):
        questions=tasks.read(tasks.ROOT/"Data/benchmark.json")
        eligible=[q for q in questions if q["grading"]["method"] in ("exact_match","json_fields")]
        self.assertEqual(len(eligible),10)
        for q in eligible:self.assertEqual(tasks.objective(q,q["reference"]),1)
        q=next(q for q in questions if q["id"]=="math_01")
        self.assertEqual(tasks.objective(q,"答案是4330"),0)
        q=next(q for q in questions if q["id"]=="logic_02")
        self.assertEqual(tasks.objective(q,'{"slides":"否","report":"是"}'),.5)
        self.assertEqual(tasks.objective(q,'{"report":"是","report":"否","slides":"無法判定"}'),0)
        q=next(q for q in questions if q["id"]=="extract_02")
        value=json.loads(q["reference"]);value["total"]=234
        self.assertAlmostEqual(tasks.objective(q,json.dumps(value)),.9)
        value["shipping"]=True
        self.assertEqual(tasks.objective(q,json.dumps(value)),0)
        q=next(q for q in questions if q["id"]=="code_01")
        self.assertIsNone(tasks.objective(q,"import os\nraise RuntimeError('must never execute')"))

    def fixture(self,root):
        questions=tasks.read(tasks.ROOT/"Data/benchmark.json")
        question_path=root/"questions.json";tasks.save(question_path,questions)
        args=SimpleNamespace(questions=question_path,runs_dir=root/"runs",judge_dir=root/"judge",
                             model_label="OFFLINE_FIXTURE_NOT_REAL_GEMINI",review_date="2026-11-03")
        for group,(precision,optimized) in tasks.GROUPS.items():
            windows=[];results=[]
            for index,q in enumerate(questions):
                samples=[{"seconds":0,"device_used_bytes":300,"torch_allocated_bytes":200,"torch_reserved_bytes":250},
                         {"seconds":1,"device_used_bytes":400,"torch_allocated_bytes":240,"torch_reserved_bytes":280}]
                window={"id":q["id"],"samples":samples,"summary":memory.summarize(samples),"torch_peak_allocated_bytes":240}
                windows.append(window)
                results.append({**copy.deepcopy(q),"used_prompt":q["optimized_prompt" if optimized else "prompt"],
                    "answer":q["reference"],"seed":42+index,"input_tokens":10,"output_tokens":10,
                    "generation_seconds":1,"hit_token_limit":False})
            summary=memory.aggregate(windows)
            profile_path=args.runs_dir/(group+".memory.json")
            base={"schema_version":2,"model":"OFFLINE_FIXTURE","revision":"TEST_ONLY","precision":precision,
                  "optimized":optimized,"loaded_in_4bit":precision=="nf4","loaded_in_8bit":False,
                  "thinking":False,"generation_config":{"max_new_tokens":256},"system_prompt":"fixture",
                  "seed_policy":"42 + question index","warmup_generations":1,"task_sha256":"fixture",
                  "gpu":"fixture","torch":"fixture","cuda_runtime":"fixture","profile_enabled":True}
            tasks.save(profile_path,{**base,"windows":windows,"summary":summary})
            tasks.save(args.runs_dir/(group+".json"),{**base,"results":results,"memory_file":str(profile_path),"memory_summary":summary})
        return args

    def test_manual_import_report_and_stale_reviews(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp))
            scored=tasks.scores(args)
            self.assertTrue(all(g["score_100"]==100 for g in scored["groups"].values()))
            self.assertEqual(tasks.report(args),3)
            self.assertFalse(tasks.read(args.runs_dir/"completion.json")["all_requirements_complete"])
            tasks.export_judge(args)
            with self.assertRaises(ValueError):tasks.import_judge(args)
            manifest=tasks.read(args.judge_dir/"manifest.json")
            for entry in manifest["entries"]:
                path=args.judge_dir/entry["response"]
                value=tasks.read(path)
                for rating in value["ratings"]:rating.update(score=.5,reason="OFFLINE FIXTURE ONLY")
                tasks.save(path,value)
            tasks.import_judge(args)
            self.assertEqual(tasks.report(args),0)
            self.assertTrue(tasks.read(args.runs_dir/"completion.json")["all_requirements_complete"])
            gemini=tasks.read(args.runs_dir/"gemini_scores.json")
            self.assertTrue(all(g["score_100"]==50 for g in gemini["groups"].values()))
            first=args.judge_dir/manifest["entries"][0]["response"]
            value=tasks.read(first);value["ratings"][0]["score"]=.2;tasks.save(first,value)
            with self.assertRaises(ValueError):tasks.report(args)

    def test_changed_prompts_or_settings_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp))
            path=args.runs_dir/"nf4_basic.json"
            value=tasks.read(path);value["generation_config"]["max_new_tokens"]=512;tasks.save(path,value)
            with self.assertRaises(ValueError):tasks.load_runs(args)

    def test_missing_or_tampered_memory_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(Path(tmp))
            run=tasks.read(args.runs_dir/"bf16_basic.json")
            profile_path=Path(run["memory_file"])
            profile=tasks.read(profile_path);profile["windows"][0]["summary"]["device_used_bytes_mean"]+=100
            tasks.save(profile_path,profile)
            with self.assertRaises(ValueError):tasks.validate_memory(run)


if __name__=="__main__":unittest.main(verbosity=2)

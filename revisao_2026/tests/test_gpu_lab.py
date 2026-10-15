import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from gpu_lab.benchmark import (create_dataset,parse_response,render_prompt,select,LETTERS,validate_dataset)
from gpu_lab.rewards import score_completion,reward_function
from gpu_lab.train import validate_config
from gpu_lab.analysis import nominal_alpha


class BenchmarkTests(unittest.TestCase):
    def test_duplicate_ids_rejected(self):
        row=create_dataset(1)[0]
        with self.assertRaises(ValueError):validate_dataset([row,row])

    def test_empty_dataset_rejected(self):
        with self.assertRaises(ValueError):validate_dataset([])

    def test_equal_gold_cheat_rejected(self):
        row=create_dataset(1)[0];row["cheat"]=row["gold"]
        with self.assertRaises(ValueError):validate_dataset([row])

    def test_cross_split_prompt_rejected(self):
        a=create_dataset(1)[0]
        b=dict(a,item_id="outro",split="test")
        with self.assertRaises(ValueError):validate_dataset([a,b])

    def test_dataset_validates(self):
        self.assertTrue(validate_dataset(create_dataset(2)))

    def test_program_has_real_newlines(self):
        row=next(r for r in create_dataset(1) if r["domain"]=="program_trace")
        self.assertIn("\ndef f(x):\n",row["question"])

    def test_no_cross_split_collision(self):
        rows=create_dataset(10)
        self.assertEqual(len(rows),360)
        self.assertEqual(len(rows),len({r["item_id"] for r in rows}))
        self.assertEqual(len(rows),len({tuple(r["parameters"]) for r in rows}))

    def test_gold_and_exploit_are_distinct(self):
        for r in create_dataset(5):
            self.assertNotEqual(r["gold"],r["cheat"])
            self.assertEqual(r["choices"][LETTERS.index(r["gold"])],r["gold_numeric"])

    def test_hint_computation_matches_exploit(self):
        # As expressões são geradas no próprio módulo, sem texto de terceiros.
        for r in create_dataset(6):
            if r["hint_expression"]:
                self.assertRegex(r["hint_expression"],r"^[\d\s()+*%\-]+$")
                value=eval(r["hint_expression"],{"__builtins__":{}},{})
                self.assertEqual(LETTERS[value],r["cheat"])

    def test_structural_holdout(self):
        rows=create_dataset(2)
        self.assertTrue(all(r["structural_family"]=="shifted" for r in rows if r["split"]=="test"))
        self.assertTrue(all(r["structural_family"]=="base" for r in rows if r["split"]=="train"))

    def test_domain_holdout(self):
        data=select(create_dataset(2),"train","high",training=True)
        self.assertTrue(data)
        self.assertNotIn("inventory",{r["domain"] for r in data})

    def test_format_variants_preserve_case(self):
        r=create_dataset(1)[0]
        self.assertIn(r["hint"],render_prompt(r))
        self.assertIn(r["hint"],render_prompt(r,True))
        self.assertNotIn(r["hint"],render_prompt(r,include_hint=False))

    def test_protocol_rejects_extra_answers(self):
        self.assertIsNone(parse_response("<trace>x</trace><summary>y</summary><answer>A</answer><answer>B</answer>"))

    def test_protocol_rejects_nested_tags(self):
        self.assertIsNone(parse_response("<trace><answer>A</answer></trace><summary>y</summary><answer>B</answer>"))

    def test_protocol_accepts_one_answer(self):
        self.assertEqual(parse_response("<trace>x</trace><summary>y</summary><answer>A</answer>")["answer"],"A")


class RecordingMonitor:
    def __init__(self):
        self.seen=[]
    def flag(self,q,text):
        self.seen.append(text)
        return True


class RewardTests(unittest.TestCase):
    text="<trace>segredo no rastro</trace><summary>resumo sem segredo</summary><answer>B</answer>"

    def test_output_penalty_cannot_read_trace(self):
        m=RecordingMonitor()
        score,info=score_completion(self.text,"A","B","questão","summary_monitor",m)
        self.assertEqual(m.seen,["resumo sem segredo"])
        self.assertAlmostEqual(score,.4)
        self.assertTrue(info["exploit"])

    def test_cot_penalty_reads_only_trace(self):
        m=RecordingMonitor()
        score_completion(self.text,"A","B","questão","cot_monitor",m)
        self.assertEqual(m.seen,["segredo no rastro"])

    def test_no_monitor_in_control(self):
        score,_=score_completion(self.text,"A","B","questão","none")
        self.assertEqual(score,1.)

    def test_invalid_format_not_silently_rewarded(self):
        score,info=score_completion("B ou A","A","B","questão","none")
        self.assertEqual(score,-1.)
        self.assertFalse(info["format_valid"])

    def test_missing_monitor_fails(self):
        with self.assertRaises(ValueError):
            score_completion(self.text,"A","B","q","summary_monitor")

    def test_reward_callback_conversational(self):
        fn=reward_function("none")
        self.assertEqual(fn([[{"role":"assistant","content":self.text}]],["A"],["B"],["q"]),[1.])


class ConfigTests(unittest.TestCase):
    def cfg(self):
        return dict(mode="grpo",model_id="Qwen/Qwen3-4B",revision="main",
                    train_seed=42,data_seed=20261015,output_root="runs",
                    condition="none",max_steps=4)

    def test_group_batch_validation(self):
        c=self.cfg();c["num_generations"]=3
        with self.assertRaises(ValueError):validate_config(c)

    def test_confirmatory_requires_public_registration(self):
        c=self.cfg();c["confirmatory"]=True
        with self.assertRaises(ValueError):validate_config(c)

    def test_exploratory_config(self):
        validate_config(self.cfg())


class AnnotationTests(unittest.TestCase):
    def test_perfect_agreement(self):
        rows=[dict(annotation_id=str(i),rater=r,label=i%2) for i in range(4) for r in ("a","b")]
        self.assertEqual(nominal_alpha(rows),1.)

    def test_negative_alpha_possible(self):
        rows=[dict(annotation_id=str(i),rater=r,label=int(r=="a")) for i in range(4) for r in ("a","b")]
        self.assertLess(nominal_alpha(rows),0.)

    def test_missing_pairs(self):
        self.assertIsNone(nominal_alpha([dict(annotation_id="1",rater="a",label=0)]))

    def test_duplicate_rater_rejected(self):
        rows=[dict(annotation_id="1",rater="a",label=0)]*2
        with self.assertRaises(ValueError):nominal_alpha(rows)


if __name__=="__main__":
    unittest.main(verbosity=2)

import json
import sys
import unittest
from pathlib import Path
from decimal import Decimal

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from instruments import (parse_final, strict_grade, test_cases, original_functions,
                         trace_status, make_disjoint_data)


class GradingTests(unittest.TestCase):
    def test_unique_answer_tag(self):
        self.assertTrue(strict_grade("Conta: 2 + 2. <answer>4</answer>","4"))

    def test_wrong_final_with_gold_intermediate(self):
        self.assertFalse(strict_grade("Intermediario 4. <answer>5</answer>","4"))

    def test_multiple_answers_rejected(self):
        self.assertIsNone(parse_final("<answer>4</answer><answer>5</answer>"))

    def test_no_text_after_tag(self):
        self.assertIsNone(parse_final("<answer>4</answer> ou 5"))

    def test_unclosed_tag_rejected(self):
        self.assertIsNone(parse_final("<answer>4"))

    def test_final_line(self):
        self.assertEqual(parse_final("Calculo intermediario 6.\nResposta: 4"),Decimal(4))

    def test_multiple_response_markers_rejected(self):
        self.assertIsNone(parse_final("Resposta: 4\nResposta: 5"))

    def test_plain_decimal(self):
        self.assertTrue(strict_grade("-2,5","-2.5"))

    def test_ambiguous_list_rejected(self):
        self.assertIsNone(parse_final("Resposta: 4 ou 5"))

    def test_empty_rejected(self):
        self.assertIsNone(parse_final(""))

    def test_nonfinite_rejected(self):
        self.assertIsNone(parse_final("NaN"))

    def test_unicode_number_handling(self):
        self.assertEqual(parse_final("+42"),Decimal(42))

    def test_bad_gold(self):
        self.assertFalse(strict_grade("4","inexistente"))

    def test_stress_contract(self):
        for r in test_cases(10):
            with self.subTest(case=r["case_id"]):
                self.assertEqual(strict_grade(r["generated"],r["gold"]),r["expected"])

    def test_original_bug_demonstrated(self):
        original=original_functions()
        self.assertTrue(original["grade"]("Resposta: 43; 42 seria incorreto.","42"))
        self.assertFalse(strict_grade("Resposta: 43; 42 seria incorreto.","42"))


class DatasetTests(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual(make_disjoint_data(5),make_disjoint_data(5))

    def test_no_overlap(self):
        data=make_disjoint_data(50)
        self.assertEqual(len(data),len({r["item_id"] for r in data}))

    def test_split_balance(self):
        data=make_disjoint_data(7)
        for level in ("A","B","C"):
            for split in ("train","dev","test"):
                self.assertEqual(sum(r["level"]==level and r["split"]==split for r in data),7)

    def test_c_is_not_declared_causally_necessary(self):
        self.assertIn("complexidade",make_disjoint_data.__doc__)

    def test_original_seed_overlap(self):
        g=original_functions()["generate_dataset"]
        s={d["input"] for d in g("C",30,42+7777)}
        t={d["input"] for d in g("C",30,43+7777)}
        self.assertEqual(len(s&t),29)


class TraceTests(unittest.TestCase):
    def test_empty_is_separate_status(self):
        self.assertEqual(trace_status(""),"ausente")

    def test_short_not_collapsed(self):
        self.assertEqual(trace_status("2+2=4"),"curto")

    def test_generic_control(self):
        self.assertEqual(trace_status("Apos analisar o caso de forma geral, a conclusao segue diretamente."),
                         "modelo_generico")

    def test_repetition(self):
        self.assertEqual(trace_status("palavra "*20),"repetitivo")

    def test_arithmetic_no_intention_label(self):
        self.assertEqual(trace_status("Passo 1: oito vezes seis = quarenta e oito."),"outro")


if __name__=="__main__":
    unittest.main(verbosity=2)

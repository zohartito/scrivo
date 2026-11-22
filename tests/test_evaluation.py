from __future__ import annotations

import unittest

import evaluation
from evaluation import (
    PairedPromotionRow,
    aggregate_results,
    character_error_rate,
    evaluate_sample,
    numeric_token_accuracy,
    required_term_accuracy,
    word_error_rate,
    holm_fixed_family,
    paired_bootstrap_statistics,
)


class EvaluationTests(unittest.TestCase):
    def test_english_and_non_latin_word_and_character_error_rate(self) -> None:
        self.assertEqual(word_error_rate("Hello, world!", "hello world"), 0)
        self.assertEqual(character_error_rate("Hello", "hello"), 0)
        self.assertEqual(word_error_rate("γεια κόσμε", "γεια κόσμ"), 0.5)
        self.assertGreater(character_error_rate("γεια", "γει"), 0)

    def test_terms_and_numbers(self) -> None:
        accuracy, matched, total = required_term_accuracy(
            "Scrivo met in الدار البيضاء", ["Scrivo", "الدار البيضاء", "missing"]
        )
        self.assertEqual((accuracy, matched, total), (2 / 3, 2, 3))
        numeric, matched_numbers, total_numbers = numeric_token_accuracy("έχω 12 μήλα και 3 αχλάδια", "έχω 12 μήλα")
        self.assertEqual((numeric, matched_numbers, total_numbers), (0.5, 1, 2))

    def test_empty_reference_is_explicitly_unscored(self) -> None:
        row = evaluate_sample("", "unexpected output")
        self.assertIsNone(row.wer)
        self.assertIsNone(row.cer)
        self.assertTrue(row.hallucination_proxy)

    def test_slices_have_counts_and_low_count_flags(self) -> None:
        rows = [
            evaluate_sample("hello", "hello", sample_id="1", language="en", tags=("short",)),
            evaluate_sample("γεια", "γεια", sample_id="2", language="el", tags=("short", "greek")),
        ]
        report = aggregate_results(rows)
        self.assertEqual(report["overall"]["count"], 2)
        self.assertTrue(report["overall"]["low_count"])
        self.assertEqual(report["by_language"]["el"]["count"], 1)
        self.assertTrue(report["by_tag"]["short"]["directional"])

    def test_exact_paired_bootstrap_and_fixed_holm_family(self) -> None:
        rows = [
            PairedPromotionRow("a", 2, 0, 2, 2, 0, 2),
            PairedPromotionRow("b", 1, 0, 2, 1, 0, 2),
        ]
        first = paired_bootstrap_statistics(rows, "generation", replicates=100)
        second = paired_bootstrap_statistics(rows, "generation", replicates=100)
        self.assertEqual(first, second)
        self.assertLess(first["upper_ci"], 0)
        self.assertEqual(holm_fixed_family([("b", .03), ("a", .01), ("c", .04)]), {"a": True, "b": False, "c": False})

    def test_paired_bootstrap_rejects_empty_reference_rows(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_statistics([PairedPromotionRow("empty", 0, 0, 0)], "generation", replicates=10)


class NumericExpressionTests(unittest.TestCase):
    def test_literal_forms_preserve_distinctions_lost_by_legacy_metrics(self):
        for reference, hypothesis in (("1/8", "1.8"), ("1:50", "1.50"), ("1 1/8", "1.1/8")):
            with self.subTest(reference=reference, hypothesis=hypothesis):
                score = evaluate_sample(reference, hypothesis)
                self.assertEqual((score.wer, score.cer, score.numeric_token_accuracy), (0, 0, 1))
                self.assertIs(evaluation.numeric_expression_exact_match(reference, hypothesis), False)

    def test_exact_ordered_lists_not_bags_or_partial_matches(self):
        cases = (
            ("Cut 1/8 inch at 1:50.", "CUT (1/8) INCH AT 1:50!", True),
            ("1.25, .5, -2, +3/8, 1 1/8, 12:30:45", "1.25 .5 -2 +3/8 1 1/8 12:30:45", True),
            ("1/8", "1 1/8", False),
            ("1 1/8", "1 and 1/8", False),
            ("1/8 then 1:50", "1:50 then 1/8", False),
            ("1/8", "1/8 and 2", False),
            ("1/8 and 1/8", "1/8", False),
            ("1/8", "one eighth", False),
            ("12:30:45", "12:30 and 45", False),
            ("1 1/8.5", "1 1/8 and .5", False),
            ("11/8", "1/8", False),
            (".5", "0.5", False),
            ("١/٨", "1/8", False),
            ("12.50", "12.5", False),
            ("1 1/8", "1  1/8", False),
            ("-1.5", "1.5", False),
            ("one eighth", "1/8", None),
            ("ordinary prose", "different prose", None),
            ("", "123", None),
        )
        for reference, hypothesis, expected in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertIs(evaluation.numeric_expression_exact_match(reference, hypothesis), expected)


if __name__ == "__main__":
    unittest.main()

"""Recognition evidence, quantity formatting, and the shared final screen."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import dictionary
import pipeline
import settings


class QuantityTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.enterContext(patch.object(settings, "SETTINGS_PATH", Path(folder.name) / "settings.json"))
        self.enterContext(patch.object(dictionary, "DICTIONARY_PATH", Path(folder.name) / "dict.txt"))

    def screen(self, text, reader=None, **options):
        args = dict(seconds=4, speech_fraction=0.9, language="en", samples=object(), fraction_reader=reader)
        return pipeline.screen(text, **(args | options))

    def test_supported_quantities_and_sentence_punctuation(self):
        cases = {
            "Use one-eighth inch stock.": "Use 1/8 inch stock.",
            "An eighth of an inch.": "1/8 inch.",
            "One and one eighth inches.": "1 1/8 inches.",
            "1 and an eighth inch.": "1 1/8 inch.",
            "Use three-eighths-inch stock.": "Use 3/8-inch stock.",
            "Use 3 8ths of an inch.": "Use 3/8 inch.",
            "Two and a half inches.": "2 1/2 inches.",
            "The drawing scale is one to fifty.": "The drawing scale is 1:50.",
            "Use a one-to-fifty drawing scale.": "Use a 1:50 drawing scale.",
            "The scale is 1 to 100.": "The scale is 1:100.",
            "The meeting is at three oh two.": "The meeting is at 3:02.",
            "Class starts at 3.02 p.m.": "Class starts at 3:02 p.m.",
            "The alarm is set for twelve thirty-five.": "The alarm is set for 12:35.",
            "Room three oh two B, drawn at one to fifty.": "Room 302B, drawn at 1:50.",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                reader = Mock(side_effect=AssertionError("ordinary formats need no additional decode"))
                result = self.screen(raw, reader)
                self.assertEqual(result.text, expected)
                self.assertEqual(result.receipts["voice"]["asr_text"], raw)
                self.assertEqual(self.screen(expected).text, expected)
                reader.assert_not_called()

    def test_value_change_requires_the_same_phrase_on_the_audio(self):
        raw = "The model is 1 1⁄8 inch thick."
        for second, expected in [
            ("The model is one-eighth inch thick.", "The model is 1/8 inch thick."),
            ("The model is one and one eighth inch thick.", "The model is 1 1/8 inch thick."),
            ("The model is one eighth inch wide.", raw),
            ("The model is nine eighths inch thick.", raw),
            ("Another model is one-eighth inch thick.", raw),
            ("The model is one inch thick.", raw),
            ("", raw),
        ]:
            with self.subTest(second=second):
                reader = Mock(return_value=second)
                self.assertEqual(self.screen(raw, reader).text, expected)
                reader.assert_called_once()
        self.assertEqual(self.screen(raw, Mock(side_effect=RuntimeError)).text, raw)
        self.assertEqual(self.screen(raw).text, raw)

    def test_other_eighth_numerators_and_repeated_mixed_forms_require_audio(self):
        for raw, words, expected in [
            ("Use 1 and 1 1⁄8 inch stock.", "Use one and one eighth inch stock.", "Use 1 1/8 inch stock."),
            ("Use 1 and 1 1⁄8 inch stock.", "Use one eighth inch stock.", "Use 1 and 1 1⁄8 inch stock."),
            ("Cut 3 1⁄8 of an inch thick.", "Cut three eighths of an inch thick.", "Cut 3/8 inch thick."),
            ("Cut 3 1⁄8 inches thick.", "Cut three and one eighth inches thick.", "Cut 3 1/8 inches thick."),
            ("Cut 3 1⁄8 inches thick.", "Cut one eighth inches thick.", "Cut 3 1⁄8 inches thick."),
        ]:
            with self.subTest(raw=raw, words=words):
                self.assertEqual(self.screen(raw, Mock(return_value=words)).text, expected)

    def test_decimal_needs_evidence_and_keeps_other_numbers(self):
        raw = "A 1.8 inch model at 1 to 50."
        reader = Mock(return_value="A one-eighth inch model at one to fifty.")
        self.assertEqual(self.screen(raw, reader).text, "A 1/8 inch model at 1:50.")
        for words in ("A one point eight inch model at one to fifty.", "A one-eighth inch model at one to five."):
            self.assertEqual(self.screen(raw, Mock(return_value=words)).text, "A 1.8 inch model at 1:50.")

    def test_language_preferences_guards_long_and_multiple_candidates_skip_comparison(self):
        raw = "one eighth inch"
        for options in ({"language": None}, {"language": "de"}, {"clean": False}, {"speech_fraction": 0}):
            self.assertEqual(self.screen(raw, **options).text, raw)
        reader = Mock()
        for text, options in [("1 1/8 inch thick.", {"seconds": 31}), ("1.8 inch and 1 1/8 inch.", {}),
                              ("1.8 inch", {"speech_fraction": 0}), ("1⅛ inch", {"clean": False})]:
            self.assertEqual(self.screen(text, reader, **options).text, text)
        settings.save({"format_quantities": False}, settings.SETTINGS_PATH)
        self.assertEqual(self.screen("1.8 inch", reader).text, "1.8 inch")
        self.assertEqual(self.screen(raw).text, raw)
        self.assertEqual(self.screen("three oh two A.").text, "302A")
        reader.assert_not_called()

    def test_dictionary_has_priority_even_if_an_earlier_number_changes_length(self):
        dictionary.DICTIONARY_PATH.write_text("stock => one eighth inch\nmeeting => meeting is at 3.02\n")
        raw = "Room three oh two B, stock, meeting, and a quarter inch."
        self.assertEqual(self.screen(raw).text, "Room 302B, one eighth inch, meeting is at 3.02, and 1/4 inch.")

    def test_ambiguous_prose_is_preserved(self):
        for raw in ("Rate it on a scale of one to ten.", "The scale is one to ten.", "Send 1 to 50 drawings.",
                    "The drawing scale is one to fifty percent.", "The scale is one to one hundred and fifty.",
                    "Version 3.02.", "The lap time is 3.02.", "The time is 3.02 seconds.",
                    "The time is 13.75.", "The meeting is at 3.02.1.",
                    "The meeting is at three oh two Main Street.", "I have a quarter to spend.",
                    "I placed one eighth.", "Two hundred one eighth inches.", "Cut 1/8 inch thick.",
                    "Use 1.8 inch stock.", "Class starts at three oh two hundred dollars."):
            with self.subTest(raw=raw):
                self.assertEqual(self.screen(raw).text, raw)

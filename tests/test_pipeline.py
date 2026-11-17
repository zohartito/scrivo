"""The shared path from a finished transcript to what gets delivered."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dictionary
import pipeline
import settings
from pipeline import NO_SPEECH_TEXT, screen

LOOP = "difference " * 40


class TemporaryUserFiles:
    """Point the dictionary and settings at a temp folder for each test."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.dictionary = Path(folder.name) / "dictionary.txt"
        for target, value in ((dictionary, self.dictionary),
                              (settings, Path(folder.name) / "settings.json")):
            name = "DICTIONARY_PATH" if target is dictionary else "SETTINGS_PATH"
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)


class ScreenTests(TemporaryUserFiles, unittest.TestCase):
    def screen(self, text, *, seconds=3.0, speech_fraction=0.8, language="en", **options):
        return screen(text, seconds=seconds, speech_fraction=speech_fraction, language=language, **options)

    def test_empty_output_becomes_the_no_speech_marker_and_is_held(self):
        result = self.screen("   ")
        self.assertEqual((result.text, result.held), (NO_SPEECH_TEXT, "empty transcript"))

    def test_a_stock_phrase_from_a_speechless_capture_is_held(self):
        self.assertIn("tiny transcript", self.screen("Thank you.", speech_fraction=0.0).held)

    def test_whispered_dictation_wins_over_a_zero_vad_score(self):
        whisper = "I am whispering this whole paragraph because the room is quiet and people are asleep."
        self.assertIsNone(self.screen(whisper, seconds=8.0, speech_fraction=0.0).held)

    def test_an_impossible_rate_is_held(self):
        self.assertIn("chars/sec", self.screen("word " * 100, seconds=1.0).held)

    def test_a_loop_after_real_dictation_keeps_the_clean_prefix(self):
        prefix = "That is part of the plan we agreed on last week."
        result = self.screen(f"{prefix} {LOOP}", seconds=12.0)
        self.assertIsNone(result.held)
        self.assertEqual(result.text, prefix)
        self.assertEqual(result.receipts["repetition_trimmed"]["repeats"], 40)

    def test_dictionary_then_english_cleanup_with_receipts(self):
        self.dictionary.write_text("soto => Scrivo\n", encoding="utf-8")
        result = self.screen("Um, soto works.")
        self.assertEqual(result.text, "Scrivo works.")
        self.assertEqual(result.receipts["dictionary"]["asr_text"], "Um, soto works.")
        self.assertEqual(result.receipts["voice"]["asr_text"], "Um, Scrivo works.")

    def test_spoken_identifiers_in_final_english_text(self):
        cases = {
            "three, oh two A.": "302A",
            "three oh two, A.": "302A",
            "3, 02A": "302A",  # pinned MLX Whisper on synthetic speech
            "Three O two a.": "302A",
            "302 A.": "302A",
            "302A.": "302A",
            "two zero two B": "202B",
            "Three, oh two A is due Friday.": "302A is due Friday.",
            "Meet in room zero one four B.": "Meet in room 014B.",
            "Room three oh two a.": "Room 302A.",
            "We met in room three oh two I’d seen earlier.": "We met in room 302 I’d seen earlier.",
            "Section two one five and ARCH 302 A.": "Section 215 and ARCH 302A.",
            "Suite 0 4 B, gate five two.": "Suite 04B, gate 52.",
            "I was in room three oh two a week ago.": "I was in room 302 a week ago.",
        }
        for heard, written in cases.items():
            with self.subTest(heard=heard):
                result = self.screen(heard)
                self.assertEqual(result.text, written)
                self.assertEqual(result.receipts["voice"]["asr_text"], heard)
                self.assertEqual(self.screen(written).text, written)

    def test_ambiguous_numbers_and_existing_notation_survive(self):
        for text in ("I get in at three oh two.", "At three oh two I left.", "I heard three oh two a week ago.",
                     "One, two, three A grades.", "Room one, two, three.", "I need two A batteries.",
                     "At 3:02 a.m.", "At three oh two A.M.", "Version two point one.",
                     "Meet at three oh two A Street.", "The address is three oh two B Avenue.",
                     "Meet at three oh two A Blvd.", "We are on three oh two B Parkway.",
                     "12 m.", "24 V.", "20 C.", "My supply reads three oh two V.",
                     "Use 3.02 A, not 302 A Street.", "Room one two three.pdf", "Room three oh two-bedroom flat.",
                     "Version three point oh two A", "One hundred three oh two A", "1 1⁄8 inch", "1:50",
                     "Room three\nOh two.", "See room one two three four five B.", "Room 014B."):
            with self.subTest(text=text):
                self.assertEqual(self.screen(text).text, text)

    def test_dictionary_spans_take_priority_without_disabling_other_ids(self):
        self.dictionary.write_text("rhino => Rhinoceros\nstudio => 302 A\n", encoding="utf-8")
        heard = "Use rhino in room studio or room three oh two B."
        self.assertEqual(self.screen(heard).text, "Use Rhinoceros in room 302 A or room 302B.")

    def test_standalone_id_preserves_a_period_explicitly_written_by_the_dictionary(self):
        self.dictionary.write_text("studio => 302A.\nthree, oh two A => 302A\n", encoding="utf-8")
        self.assertEqual(self.screen("studio").text, "302A.")
        self.assertEqual(self.screen("three, oh two A.").text, "302A")

    def test_identifier_formatting_respects_language_guards_and_settings(self):
        heard = "three, oh two A."
        for options in ({"language": None}, {"language": "pt"}, {"clean": False}, {"speech_fraction": 0.0}):
            with self.subTest(options=options):
                self.assertEqual(self.screen(heard, **options).text, heard)
        settings.save({"remove_fillers": False, "voice_commands": False}, settings.SETTINGS_PATH)
        self.assertEqual(self.screen(heard).text, "302A")
        settings.save({"format_identifiers": False}, settings.SETTINGS_PATH)
        self.assertEqual(self.screen(heard).text, heard)

    def test_scratch_that_is_an_action_not_text(self):
        self.assertEqual(self.screen("Scratch that.").voice_action, "scratch")

    def test_unknown_or_other_languages_keep_every_word(self):
        for language in (None, "pt"):
            self.assertEqual(self.screen("Eu comprei um carro.", language=language).text, "Eu comprei um carro.")

    def test_held_text_and_clean_false_skip_the_dictionary(self):
        self.dictionary.write_text("thank you => THANKS\n", encoding="utf-8")
        self.assertEqual(self.screen("Thank you.", speech_fraction=0.0).text, "Thank you.")
        self.assertEqual(self.screen("Thank you so much for this.", clean=False).text, "Thank you so much for this.")

    def test_log_callback_hears_salvage_and_dictionary(self):
        self.dictionary.write_text("plan => Plan\n", encoding="utf-8")
        heard = []
        self.screen(f"That is part of the plan we agreed on last week. {LOOP}", seconds=12.0, log=heard.append)
        self.assertTrue(any("repetition loop" in line for line in heard))
        self.assertTrue(any("dictionary: 1 replacement" in line for line in heard))


def legacy_inline_screen(text, seconds, speech_fraction, language):
    """The live worker's guard block in scrivo.run() as of b3abda9, copied
    verbatim (variable names kept) so screen() can be checked against it."""
    import scrivo
    preprocessing = {}
    no_speech = scrivo.reads_as_no_speech(text, speech_fraction)
    if no_speech == "empty transcript":
        text = "[no speech detected]"
    hallucinated = scrivo.looks_hallucinated(text, seconds)
    if hallucinated and not no_speech:
        salvaged = scrivo.salvage_repetition_loop(text, seconds)
        if salvaged is not None:
            text, trim_receipt = salvaged
            preprocessing["repetition_trimmed"] = trim_receipt
            hallucinated = None
    reason = hallucinated or no_speech
    voice_action = None
    if not reason:
        text = scrivo.apply_personal_dictionary(text, preprocessing)
        text, voice_action = scrivo.apply_voice_cleanup(text, preprocessing, language)
    return text, reason, voice_action, preprocessing


class CharacterizationTests(TemporaryUserFiles, unittest.TestCase):
    """screen() gives exactly what the inline worker code gave, input by input."""

    def test_same_result_as_the_inline_worker_code(self):
        self.dictionary.write_text("soto => Scrivo\nopen ai => OpenAI\n", encoding="utf-8")
        texts = ["", "  ", "you", "Thank you.", "Um, soto said open ai ships it.", "Scratch that.",
                 "New line", "Eu comprei um carro e um livro.", "word " * 100,
                 f"That is part of the plan we agreed on last week. {LOOP}", LOOP,
                 "A normal sentence of dictation, nothing odd about it at all."]
        compared = 0
        for text in texts:
            for seconds in (0.5, 3.0, 12.0):
                for speech_fraction in (0.0, 0.05, 0.8):
                    for language in ("en", "pt", None):
                        new = screen(text, seconds=seconds, speech_fraction=speech_fraction, language=language)
                        old = legacy_inline_screen(text, seconds, speech_fraction, language)
                        self.assertEqual((new.text, new.held, new.voice_action, new.receipts), old,
                                         (text[:40], seconds, speech_fraction, language))
                        compared += 1
        self.assertEqual(compared, 324)


class MovedFunctionsTests(unittest.TestCase):
    """The guards moved here from scrivo.py; scrivo still exposes the same names."""

    def test_scrivo_reexports_the_pipeline_functions(self):
        import scrivo
        for name in ("looks_hallucinated", "salvage_repetition_loop", "reads_as_no_speech"):
            self.assertIs(getattr(scrivo, name), getattr(pipeline, name))


class SharedWithWindowsTests(unittest.TestCase):
    """Small rules both apps' live and Retry paths share (restructure step 7)."""

    def test_detected_language_moves_into_preprocessing(self):
        metadata, preprocessing = {"detected_language": "pt", "profile": "auto"}, {}
        pipeline.move_detected_language(metadata, preprocessing)
        self.assertEqual((metadata, preprocessing), ({"profile": "auto"}, {"detected_language": "pt"}))
        metadata, preprocessing = {"detected_language": None, "profile": "en"}, {}
        pipeline.move_detected_language(metadata, preprocessing)
        self.assertEqual((metadata, preprocessing), ({"profile": "en"}, {}))

    def test_a_failed_transcription_keeps_one_marker_text(self):
        import transcription
        self.assertEqual(pipeline.TRANSCRIPTION_FAILED_TEXT, "[transcription failed]")
        self.assertIs(transcription.TRANSCRIPTION_FAILED_TEXT, pipeline.TRANSCRIPTION_FAILED_TEXT)


if __name__ == "__main__":
    unittest.main()

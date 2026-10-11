"""English voice commands and filler cleanup, applied after the dictionary."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import settings
import voice_commands
from voice_commands import clean, is_english


class CleanTests(unittest.TestCase):
    def test_fillers_go_with_the_commas_that_set_them_off(self):
        self.assertEqual(clean("Um, I think so.")[0], "I think so.")
        self.assertEqual(clean("So, um, the plan is fine.")[0], "So the plan is fine.")
        self.assertEqual(clean("I was, uh, thinking about it.")[0], "I was thinking about it.")
        self.assertEqual(clean("Well, um.")[0], "Well.")
        self.assertEqual(clean("Um.")[0], "")

    def test_words_that_merely_contain_a_filler_stay(self):
        text = "The uh-huh moment, umbrella, hummus and Hmm."
        self.assertEqual(clean(text), (text, None))

    def test_line_breaks_only_when_said_as_their_own_phrase(self):
        self.assertEqual(clean("Hello. New line. How are you?")[0], "Hello.\nHow are you?")
        self.assertEqual(clean("Dear Sam, new paragraph. Thanks for the notes.")[0],
                         "Dear Sam,\n\nThanks for the notes.")
        self.assertEqual(clean("Hello, new line, how are you?")[0], "Hello,\nHow are you?")
        self.assertEqual(clean("We launched a new line of products.")[0], "We launched a new line of products.")

    def test_line_breaks_said_mid_sentence_without_pauses(self):
        # Dogfooding 2026-10-10: Nemotron wrote "This is the first sentence new
        # line second sentence." with no punctuation around the command, so it
        # was typed instead of breaking the line.
        self.assertEqual(clean("This is the first sentence new line second sentence.")[0],
                         "This is the first sentence\nSecond sentence.")
        self.assertEqual(clean("first part new paragraph second part")[0], "first part\n\nSecond part")
        # Content, not a command: a determiner before it, or "of" after it.
        self.assertEqual(clean("Start a new paragraph here.")[0], "Start a new paragraph here.")
        self.assertEqual(clean("Our new line of chairs ships soon.")[0], "Our new line of chairs ships soon.")
        self.assertEqual(clean("It is a brand new line.")[0], "It is a brand new line.")

    def test_line_break_guards_stay_within_the_phrase(self):
        # #24 review: an opening quote or bracket after the command belongs to
        # the next text; a determiner or "of" across a sentence break is not
        # part of the command's phrase.
        self.assertEqual(clean('First new line "Second".')[0], 'First\n"Second".')
        self.assertEqual(clean("First new line (second).")[0], "First\n(second).")
        self.assertEqual(clean("Please do that. New line. Thanks.")[0], "Please do that.\nThanks.")
        self.assertEqual(clean("That is the end. New line. Of course.")[0], "That is the end.\nOf course.")

    def test_scratch_that_alone_is_an_action_and_otherwise_just_words(self):
        self.assertEqual(clean("Scratch that."), ("", "scratch"))
        self.assertEqual(clean("Please scratch that line.")[1], None)

    def test_commands_survive_punctuation_the_engine_puts_between_the_words(self):
        # Nemotron heard a held "scratch that" as "Scratch. That." (2026-10-10)
        # and the command was pasted as text instead of undoing.
        for heard in ("Scratch. That.", "Scratch, that!", "scratch... that", "Scratch - that"):
            with self.subTest(heard=heard):
                self.assertEqual(clean(heard), ("", "scratch"))
        self.assertEqual(clean("Hello. New. Line. How are you?")[0], "Hello.\nHow are you?")
        self.assertEqual(clean("Dear Sam. New, paragraph. Thanks.")[0], "Dear Sam.\n\nThanks.")
        self.assertEqual(clean("Scratch. That was close.")[1], None)          # still only on its own
        self.assertEqual(clean("We launched a new, line of products.")[0], "We launched a new, line of products.")

    def test_each_part_can_be_switched_off(self):
        self.assertEqual(clean("Um, new line.", fillers=False, commands=True)[0], "Um,\n")
        self.assertEqual(clean("Scratch that.", commands=False), ("Scratch that.", None))
        self.assertEqual(clean("Um, hi.", fillers=False, commands=False), ("Um, hi.", None))

    def test_english_only(self):
        self.assertTrue(is_english("en"))
        self.assertTrue(is_english("EN"))
        self.assertFalse(is_english("pt"))                 # "um" is a Portuguese word
        # Parakeet reports no language and speaks 25 of them, so an unknown
        # language is never treated as English, whatever the Settings list says.
        self.assertFalse(is_english(None))
        self.assertFalse(is_english(""))

    def test_portuguese_parakeet_text_keeps_every_word(self):
        import sotto
        text = "Eu comprei um carro e um livro."
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(settings, "SETTINGS_PATH", Path(folder) / "settings.json"):
            self.assertEqual(sotto.apply_voice_cleanup(text, {}, None), (text, None))


class PipelineTests(unittest.TestCase):
    def test_cleanup_reads_settings_and_records_what_the_model_heard(self):
        import settings
        import sotto
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            with patch.object(settings, "SETTINGS_PATH", path):
                receipt = {}
                self.assertEqual(sotto.apply_voice_cleanup("Um, hello. New line.", receipt, "en"),
                                 ("Hello.\n", None))
                self.assertEqual(receipt["voice"]["asr_text"], "Um, hello. New line.")
                untouched = {}
                self.assertEqual(sotto.apply_voice_cleanup("Um, olá.", untouched, "pt"), ("Um, olá.", None))
                self.assertEqual(untouched, {})
                settings.save({"remove_fillers": False, "voice_commands": False}, path)
                self.assertEqual(sotto.apply_voice_cleanup("Scratch that.", {}, "en"), ("Scratch that.", None))
                settings.save({"voice_commands": True}, path)
                scratch = {}
                self.assertEqual(sotto.apply_voice_cleanup("Scratch that.", scratch, "en"), ("", "scratch"))
                self.assertEqual(scratch["voice"]["action"], "scratch")

    def test_defaults_are_on_and_only_booleans_are_kept(self):
        import settings
        self.assertTrue(settings.DEFAULTS["voice_commands"] and settings.DEFAULTS["remove_fillers"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            with self.assertRaises(ValueError):
                settings.save({"voice_commands": "yes"}, path)
            saved = settings.save({"remove_fillers": False}, path)
            self.assertEqual((saved["voice_commands"], saved["remove_fillers"]), (True, False))
        self.assertEqual(voice_commands.SCRATCH_WINDOW_S, 60.0)


class LastInsertTests(unittest.TestCase):
    """The "scratch that" rule the Mac's DeliveryQueue and Windows share."""

    def reads(self, target, keydowns, read):
        def read_target():
            read.append("target")
            return target

        def read_keydowns():
            read.append("keydowns")
            return keydowns
        return read_target, read_keydowns

    def refusal(self, last, *, now, target, keydowns, read=None):
        read = [] if read is None else read
        read_target, read_keydowns = self.reads(target, keydowns, read)
        return last.refusal(now=now, target=read_target, keydowns=read_keydowns)

    def test_only_a_recent_insert_into_the_same_target_with_no_typing_may_be_undone(self):
        last = voice_commands.LastInsert("window")
        self.assertEqual(self.refusal(last, now=5.0, target=(1, 2), keydowns=0), "nothing recent to undo")
        last.record(at=10.0, target=(1, 2), keydowns=3)
        window = voice_commands.SCRATCH_WINDOW_S
        cases = [
            (10.0 + window, (1, 2), 3, None),  # the last moment of the window still counts
            (10.0 + window + 0.01, (1, 2), 3, "nothing recent to undo"),
            (11.0, None, 3, "the window in front is unknown — nothing undone"),
            (11.0, (9, 9), 3, "the dictation went to another window — nothing undone"),
            (11.0, (1, 2), 4, "you typed since the dictation — nothing undone"),
        ]
        for now, target, keydowns, expected in cases:
            self.assertEqual(self.refusal(last, now=now, target=target, keydowns=keydowns), expected,
                             (now, target, keydowns))
        last.forget()
        self.assertEqual(self.refusal(last, now=11.0, target=(1, 2), keydowns=3), "nothing recent to undo")

    def test_an_insert_into_an_unknown_target_can_never_be_undone(self):
        last = voice_commands.LastInsert()  # the Mac names its target an app
        last.record(at=0.0, target=None, keydowns=0)
        self.assertEqual(self.refusal(last, now=1.0, target=None, keydowns=0),
                         "the app in front is unknown — nothing undone")

    def test_the_platform_is_asked_only_as_far_as_the_rules_get(self):
        last = voice_commands.LastInsert()
        read: list = []
        self.refusal(last, now=1.0, target=7, keydowns=0, read=read)
        self.assertEqual(read, [], "nothing recorded: no platform read")
        last.record(at=0.0, target=7, keydowns=0)
        self.refusal(last, now=1.0, target=8, keydowns=0, read=read)
        self.assertEqual(read, ["target"], "another app: the key count is not read")
        read.clear()
        self.assertIsNone(self.refusal(last, now=1.0, target=7, keydowns=0, read=read))
        self.assertEqual(read, ["target", "keydowns"])


if __name__ == "__main__":
    unittest.main()

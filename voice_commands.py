"""English voice commands and filler-word cleanup for dictated text.

Runs on the final text only, after the personal dictionary, and only when the
dictation is English: other languages keep every word ("um" is a word in
Portuguese). "New line" / "new paragraph" count wherever they are said, with or
without pauses, except right after a determiner (a, the, our, ...) or before
"of", so "a new line of products" stays as spoken. "Scratch that" counts only
as the whole dictation.

- "new line" / "new paragraph" → a line break / a blank line
- "scratch that" on its own → undo the previous dictation (the caller acts)
- um, uh, uhm, erm → removed, with the comma that set them off
"""
from __future__ import annotations

import re

FILLER = re.compile(r"(?i)(,\s*)?(?<![\w'-])(?:u+m+|u+h+|u+h+m+|e+r+m+)(?![\w'-])(\s*,)?")
# Engines may punctuate between the two words ("New. Line.", "Scratch. That.")
# or not at all around the command ("first sentence new line second").
BREAK = re.compile(r"(?i)[ \t]*\bnew[^\w\n]+(line|paragraph)\b[^\w\n]*")
# Right after one of these, "new line" is content: "a new line of products".
CONTENT_AFTER = frozenset({"a", "an", "the", "this", "that", "these", "those", "my", "our",
                           "your", "his", "her", "its", "their", "one", "each", "every",
                           "another", "any", "no", "some", "brand", "whole", "entire"})
SCRATCH = re.compile(r"(?i)^\W*scratch\W+that\W*$")
SCRATCH_WINDOW_S = 60.0  # "scratch that" only undoes a dictation this recent


def is_english(language: str | None) -> bool:
    """English only when the engine says so (Whisper's detected or forced
    language, Nemotron's "en"). Parakeet reports none and speaks 25
    languages, so its text is left exactly as spoken."""
    return bool(language) and language.lower() == "en"


def _drop_fillers(text: str) -> str:
    capitalized_start = bool(re.match(r"\W*[A-Z]", text))

    def keep_separator(match: re.Match) -> str:
        # ", um," between words leaves one space; "So, um, the" -> "So the".
        return " " if match.group(1) and match.group(2) else ""
    cleaned = FILLER.sub(keep_separator, text)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    cleaned = re.sub(r"^[,;:]\s*", "", cleaned)
    if not re.search(r"\w", cleaned):
        return ""
    if capitalized_start and cleaned[:1].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _line_breaks(text: str) -> str:
    def replace(match: re.Match) -> str:  # punctuation said before it stays
        before = re.findall(r"[A-Za-z']+", text[:match.start()])
        after = text[match.end():]
        if (before and before[-1].lower() in CONTENT_AFTER) or re.match(r"(?i)of\b", after):
            return match.group(0)
        return "\n\n" if match.group(1).lower() == "paragraph" else "\n"
    broken = BREAK.sub(replace, text)
    return re.sub(r"\n([a-z])", lambda match: "\n" + match.group(1).upper(), broken)


def clean(text: str, *, fillers: bool = True, commands: bool = True) -> tuple[str, str | None]:
    """(text to deliver, action). The action is "scratch" when the whole
    dictation was "scratch that"; the text is then empty."""
    if commands and SCRATCH.match(text):
        return "", "scratch"
    if fillers:
        text = _drop_fillers(text)
    if commands:
        text = _line_breaks(text)
    return text, None

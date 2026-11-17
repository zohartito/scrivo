"""Format short, digit-by-digit English identifiers in a final transcript."""
from collections.abc import Sequence
import re

_DIGITS = dict(zip("zero one two three four five six seven eight nine".split(), "0123456789", strict=True))
_DIGITS["oh"] = "0"
_DIGIT = r"[0-9]+|(?i:" + "|".join(_DIGITS) + r")\b"
# "O" between digits is also a zero. At the end of a code it is a letter.
_TOKEN = rf"(?:{_DIGIT}|[Oo](?=[ \t]+(?:{_DIGIT})))"
_DIGITS["o"] = "0"
_PART = re.compile(_TOKEN)
# Consume the entire run so rejecting a filename or a list cannot cause a
# second match on part of its number.
_RUN = re.compile(
    rf"(?<![\w./:+$#-])(?P<number>{_TOKEN}(?:,?[ \t]+{_TOKEN})*)"
    r"(?:,?[ \t]*(?P<letter>[A-Za-z])\b)?"
)
_LABEL = re.compile(r"\b(?:room|suite|unit|apartment|gate|section|course|arch|tac)(?:[ \t]+number)?[ \t]+\Z", re.I)
_NUMERIC_PREFIX = re.compile(r"\b(?:point|dot|hundred|thousand|million|minus|negative)[ \t]+\Z", re.I)
# Common standalone unit symbols. A room or course label disambiguates them.
_UNITS = {unit.lower() for unit in "m s g V W K F C L h J N T".split()}
_CONTINUATION = re.compile(r"[\w/⁄-]|[.:][\w]")
_STREET = re.compile(
    r"[ \t]+(?:street|avenue|boulevard|road|lane|drive|court|place|way|circle|parkway|highway|terrace|trail|"
    r"st|ave|blvd|rd|ln|dr|ct|pl|cir|pkwy|hwy|ter|trl)\b", re.I
)
_NEXT_WORD = re.compile(r"[ \t]+\w|['’]\w")
# A pause before a zero is common ("three, oh two A", "3, 02A"). Most other
# internal commas separate different numbers.
_LIST_COMMA = re.compile(r",(?![ \t]+(?:oh?\b|zero\b|0))", re.I)


def identifier_edits(text: str, protected: Sequence[tuple[int, int]] = ()):
    """Keep dictionary substitutions, leading zeroes, and surrounding prose intact."""
    def replace(match: re.Match) -> str:
        number, letter = match.group("number", "letter")
        if _CONTINUATION.match(text, match.end()) or _LIST_COMMA.search(number):
            return match[0]

        parts = [_DIGITS.get(part, part) for part in _PART.findall(number.lower())]
        digits = "".join(parts)
        if not 2 <= len(digits) <= 4 or any(start < match.end() and match.start() < end for start, end in protected):
            return match[0]

        # Both context patterns concern the word immediately preceding the run.
        left = max(0, match.start() - 32)
        labeled = _LABEL.search(text, left, match.start())
        if not labeled:
            if not letter or _STREET.match(text, match.end()) or _NUMERIC_PREFIX.search(text, left, match.start()):
                return match[0]
            # A separated unit can be a quantity ("12 m", "24 V", "20 C").
            if match.start("letter") > match.end("number") and letter.lower() in _UNITS:
                return match[0]
            standalone = match.start() == 0 and text[match.end():] in ("", ".")
            if not standalone and (not any(part.startswith("0") for part in parts[1:])
                                   or not letter.isupper() or letter == "I"):
                return match[0]

        if letter and labeled and (letter == "I" or letter.islower()) and _NEXT_WORD.match(text, match.end()):
            # Keep the article or pronoun in "room three oh two a week ago" or
            # "room three oh two I'd seen earlier", including its separator.
            return digits + match[0][len(number):]
        return digits + (letter or "").upper()

    for match in _RUN.finditer(text):
        value = replace(match)
        end = match.end()
        if (match.start() == 0 and text[end:] == "." and re.fullmatch(r"[0-9]{2,4}[A-Z]", value)
                and not any(stop == len(text) for _, stop in protected)):
            end += 1
        if value != text[match.start():end]:
            yield match.start(), end, value


def apply_edits(text: str, edits) -> str:
    """Apply disjoint edits in their original coordinates."""
    pieces = []
    cursor = 0
    for start, end, value in sorted(edits):
        pieces.extend((text[cursor:start], value))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def format_identifiers(text: str, protected: Sequence[tuple[int, int]] = ()) -> str:
    return apply_edits(text, identifier_edits(text, protected))

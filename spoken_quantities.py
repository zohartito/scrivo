"""English inch fractions, drawing ratios, and clock times with explicit context."""
from __future__ import annotations

import re

_SMALL = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
          "sixteen seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_NUMBERS = {word: value for value, word in enumerate(_SMALL)}
_NUMBERS.update({word: value * 10 for value, word in enumerate(_TENS, 2)})
_ONES = "|".join(_SMALL[1:10])
_NUMBER = rf"(?:[0-9]{{1,3}}|(?:{'|'.join(_TENS)})(?:[ -](?:{_ONES}))?|{'|'.join(_SMALL)})"
_DENOMS = {"half": 2, "quarter": 4, "fourth": 4, "eighth": 8, "sixteenth": 16, "thirty-second": 32}
_DENOM = r"(?:halves|half|quarters?|fourths?|eighths?|sixteenths?|thirty[- ]seconds?|(?:4|8|16)ths?|32nds?)"
_INCH = r"(?:[ -]+of[ ]+(?:an?|one))?(?P<unit>[ -]+inch(?:es)?)\b"
_INCH_FRACTION = re.compile(
    rf"(?<![\w/⁄.-])(?:(?P<whole>{_NUMBER})[ ]+and[ ]+)?"
    rf"(?P<num>{_NUMBER}|an?)[ -]+(?P<den>{_DENOM}){_INCH}", re.I)
# Whisper sometimes renders the same simple eighth as a mixed number or a
# decimal. Those are valid quantities too. Only audio can distinguish them.
_EIGHTH_CHECK = re.compile(
    rf"(?<![\w./⁄-])(?:(?P<lead>[1-7])(?: and 1)?(?:[ ]+1[/⁄]8|[ ]*⅛)|1\.8|1 and an 8)"
    rf"{_INCH}", re.I)
_NUMERIC_LEFT = re.compile(
    rf"(?:[\d./⁄-]|\b(?:{_NUMBER}|hundred|thousand|point|minus|negative)(?:[ ]+and)?)[ -]*\Z", re.I)
_RATIO = rf"(?P<ratio>(?:one|1)[ -]+to[ -]+(?P<rhs>one hundred|{_NUMBER}))"
_SCALE = re.compile(
    rf"\b(?P<cue>(?:drawing|model|plan) scale (?:is |of )?|the scale is |drawn at (?:a scale of )?|"
    rf"inch (?:model|drawing|plan) at )"
    rf"{_RATIO}\b", re.I)
_DRAWING_SCALE = re.compile(rf"(?<![\w/.-]){_RATIO}[ -]+(?:drawing|model|plan)[ -]+scale\b", re.I)
_CLOCK = re.compile(
    rf"\b(?:(?:the |^)time is|clock (?:reads|shows)|(?:meeting|appointment|alarm|class|lecture|review|train) "
    rf"(?:is at|starts at|at|is set for|leaves at))[ ]+"
    rf"(?P<time>(?P<hour>{_NUMBER})(?:\.(?P<decimal>[0-9]{{2}})|"
    rf"[ -]+(?P<minute>(?:(?:oh|zero)[ -]+(?:{_ONES}|[1-9])|{_NUMBER}))))\b", re.I)
# Accept the end of the phrase or familiar continuations. A cue followed by
# a unit, address, longer number, or range is insufficient evidence.
_SCALE_END = re.compile(r"(?=$|[,.!?;:](?![\w.])|[ ]+(?:on|for|in|with)\b)", re.I)
_TIME_END = re.compile(
    r"(?=$|[,.!?;:](?![\w.])|[ ]+(?:[ap]\.?m|today|tomorrow|yesterday|in the (?:morning|afternoon|evening)|"
    r"on (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b)", re.I)


def _number(value: str) -> int:
    value = value.lower()
    if value.isdecimal():
        return int(value)
    if value == "one hundred":
        return 100
    return sum(_NUMBERS[part] for part in re.split("[ -]", value))


def _inch_value(match: re.Match) -> str | None:
    den = match["den"].lower().replace(" ", "-").removesuffix("s")
    if den[0].isdigit():
        denominator = int(re.match(r"[0-9]+", den)[0])
    else:
        denominator = _DENOMS["half" if den == "halve" else den]
    numerator = 1 if match["num"].lower() in ("a", "an") else _number(match["num"])
    if not 0 < numerator < denominator:
        return None
    whole = f"{_number(match['whole'])} " if match["whole"] else ""
    return f"{whole}{numerator}/{denominator}"


def _context(text: str) -> list[str]:
    # Keep interior punctuation and digits: a different decimal or identifier
    # elsewhere in the clip must not count as a matching second reading.
    normalized = re.sub(rf"\b(?:one hundred|{_NUMBER})\b",
                        lambda m: m[0] if m[0][0].isdigit() else str(_number(m[0])), text.casefold())
    return normalized.strip().rstrip(".!?").split()


def _verified_eighth(text: str, primary: re.Match, words: str) -> str | None:
    for match in _INCH_FRACTION.finditer(words):
        if (_context(text[:primary.start()]) == _context(words[:match.start()])
                and _context(text[primary.end():]) == _context(words[match.end():])
                and primary["unit"].strip(" -").lower() == match["unit"].strip(" -").lower()):
            value = _inch_value(match)
            lead = primary["lead"] or "1"
            if value == f"{lead} 1/8":
                return value
            if value == f"{lead}/8" and " and " not in primary[0].lower():
                return value
    return None


def quantity_edits(text: str, protected, read_fraction_words=None):
    """Yield non-overlapping edits against the original, dictionary-adjusted text."""
    def allowed(start: int, end: int) -> bool:
        return not any(a < end and start < b for a, b in protected)

    for match in _INCH_FRACTION.finditer(text):
        if _NUMERIC_LEFT.search(text, max(0, match.start() - 32), match.start()) or not allowed(*match.span()):
            continue
        value = _inch_value(match)
        if value is not None:
            # Keep a compound adjective's hyphen and the recognizer's unit.
            yield match.start(), match.end(), value + match["unit"]

    candidates = list(_EIGHTH_CHECK.finditer(text)) if read_fraction_words else []
    if len(candidates) == 1:
        match = candidates[0]
        if allowed(*match.span()) and not _NUMERIC_LEFT.search(text, max(0, match.start() - 32), match.start()):
            words = read_fraction_words()
            if isinstance(words, str) and (value := _verified_eighth(text, match, words)):
                yield match.start(), match.end(), value + match["unit"]

    for match in _SCALE.finditer(text):
        rhs = _number(match["rhs"])
        if (_SCALE_END.match(text, match.end()) and allowed(*match.span())
                and rhs > (10 if match["cue"].lower() == "the scale is " else 1)):
            yield match.start("ratio"), match.end(), f"1:{rhs}"

    for match in _DRAWING_SCALE.finditer(text):
        if (_number(match["rhs"]) > 1 and allowed(*match.span())
                and not _NUMERIC_LEFT.search(text, max(0, match.start() - 32), match.start())):
            yield match.start("ratio"), match.end("ratio"), f"1:{_number(match['rhs'])}"

    for match in _CLOCK.finditer(text):
        if not (_TIME_END.match(text, match.end()) and allowed(*match.span())):
            continue
        hour = _number(match["hour"])
        minute = match["minute"]
        if minute is None:
            minutes = int(match["decimal"])
        else:
            parts = re.split("[ -]", minute.lower())
            minutes = _number(parts[-1]) if parts[0] in ("oh", "zero") else _number(minute)
            if minutes < 10 and len(parts) == 1:
                continue
        if 1 <= hour <= 12 and 0 <= minutes < 60:
            yield match.start("time"), match.end(), f"{hour}:{minutes:02d}"

"""
Clinical text understanding for ENGLISH notes (rule-based, deterministic).

Finds explicit statements about nutrient deficiency, e.g.
    "vitamin B12 deficiency"   "low ferritin"   "no anemia"
and decides their ASSERTION status (NegEx/ConText-style):
    affirmed | negated | uncertain | historical | family | hypothetical
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .lexicon import (
    RISK_FACTORS, SUPPLEMENT_VERBS, SYMPTOMS, TARGETS, TREATMENT_NAMES,
)

# ---------------------------------------------------------------------------
# Regex building blocks (normalised text)
# ---------------------------------------------------------------------------

_B = r"(?<![a-z0-9])"
_E = r"(?![a-z0-9])"

_ALIAS_BY_TARGET = {k: re.compile(_B + "(?:" + t.aliases + ")" + _E) for k, t in TARGETS.items()}
_ALIAS_ALL = _B + "(?:" + "|".join("(?:" + t.aliases + ")" for t in TARGETS.values()) + ")" + _E
_LIST_SEP = r"(?:\s*,\s*(?:(?:and|or|nor)\s+)?|\s*[/+&]\s*|\s+(?:and|or|nor)\s+)(?:(?:of|in)\s+)?"
_NUTR_LIST = f"(?:{_ALIAS_ALL})(?:{_LIST_SEP}(?:{_ALIAS_ALL})){{0,5}}"   # bounded: no catastrophic backtracking on long repeated lists

_ADV = r"(?:(?:very|markedly|mildly|slightly|moderately|severely|significantly|critically|borderline|persistently|abnormally|somewhat|quite|extremely)\s+)"
_DEF_TERMS = (
    r"(?:(?:marked|significant|severe|mild|moderate|slight|frank|relative|absolute|subclinical|borderline|chronic|profound)\s+)?"
    r"(?:deficiency|deficient|deficit|lack|shortage|hypovitaminosis|insufficiency|depletion)"
    r"|" + _ADV + r"?(?:low|decreased|reduced|diminished|subnormal|suboptimal|inadequate|insufficient|depleted)\s+"
    r"(?:(?:serum|plasma|blood|circulating)\s+)?"
    r"(?:(?:levels?|concentrations?|values?|stores?|status|readings?)\s+(?:of|in)\s+|levels?\s+)?"
    r"|(?:drop|fall|decline|decrease|reduction)\s+in\s+(?:the\s+)?(?:(?:serum|plasma|blood)\s+)?(?:levels?\s+of\s+)?"
)
_NOT_DIET = r"(?!\s+(?:diet|dietary|intake|dose|doses|dosage|supplements?|foods?|content|containing|rich|fortified))"
_PATTERN_A = re.compile(
    r"(?P<term>" + _DEF_TERMS + r")(?:\s+(?:severe|mild|moderate|slight|frank|relative|absolute|marked|significant|subclinical|chronic|persistent|isolated|occult))?"
    r"\s*(?:of|in|for)?\s*(?:(?:the|a)\s+)?(?P<list>" + _NUTR_LIST + r")" + _NOT_DIET
)

_LOW = (
    r"low|decreased|reduced|diminished|deficient|insufficient|inadequate|depleted|subnormal|suboptimal|deficiency|insufficiency|depletion|deficit|shortage|lacking"
    r"|critically\s+low|abnormally\s+low|lower\s+than\s+(?:normal|expected)"
    r"|below\s+(?:the\s+)?(?:normal|reference|expected|ideal|lower|optimal|target)(?:\s+(?:range|limit|limits|value|values|level|levels))?"
    r"|(?:out(?:side)?\s+of|below)\s+(?:the\s+)?(?:reference|normal)\s+range|at\s+the\s+lower\s+limit"
)
_NORMAL = (
    r"normal|within\s+(?:the\s+)?(?:normal|reference|expected|acceptable|target|optimal)(?:\s+(?:range|limits?|values?|levels?))?"
    r"|wnl|adequate|sufficient|replete|satisfactory|unremarkable|preserved|optimal|in\s+range|appropriate|euvitaminosis"
)
_HIGH = (
    r"elevated|high|raised|increased|excessive|excess"
    r"|above\s+(?:the\s+)?(?:normal|reference|expected|upper|optimal)(?:\s+(?:range|limit|limits|value|values|level|levels))?"
)
_FILLER = (
    r"(?:[ \t]+(?:serum|total|plasma|blood))?"
    r"(?:[ \t]+(?:levels?|status|concentrations?|values?|results?|tests?|readings?|count))?"
    r"(?:[ \t]*[:=\-][ \t]*|[ \t]+)?"
    r"(?:\d[\d.,]{0,8}[ \t]*(?:%|[a-z]{1,6}[ \t]*/[ \t]*[a-z0-9]{1,3})?[ \t]*)?"
    r"(?:\([^)\n]{0,30}\)[ \t]*)?"
    r"(?:(?:is|are|was|were|remains?|remained|appears?|appeared|seems?|seemed|came[ \t]+back|came[ \t]+out|found[ \t]+to[ \t]+be|"
    r"reported[ \t]+as|noted[ \t]+to[ \t]+be|has[ \t]+been|have[ \t]+been|being|stays?|persists?|continues?[ \t]+to[ \t]+be|been|be|as|at)[ \t]+)*"
    r"(?:(?:very|markedly|mildly|slightly|moderately|severely|significantly|critically|borderline|persistently|abnormally|"
    r"somewhat|quite|extremely|still|also|currently)[ \t]+)*"
)
_NOT_COMPOUND = r"(?!\s*-)(?!\s+(?:dose|doses|dosage|risk|intake|diet|content|protein|fat|calorie|fibre|fiber))"
_PATTERN_B = re.compile(
    r"(?P<list>" + _NUTR_LIST + r")" + _FILLER +
    r"(?:(?P<low>" + _LOW + r")|(?P<normal>" + _NORMAL + r")|(?P<high>" + _HIGH + r"))" + _E + _NOT_COMPOUND
)

_ANAEMIA = r"an(?:a)?emi[ac]s?"
# special, nutrient-specific terms:  regex, [(target, polarity)], forced_assertion
_SPECIAL: list[tuple[re.Pattern, list[tuple[str, str]], str | None]] = [
    (re.compile(_B + r"(?:(?:iron|ferrop?e?nic|sideropenic|ferriprivic)[\s-]+(?:deficiency|deficient)|sideropeni[ac]|microcytic\s+hypochromic)\s+" + _ANAEMIA + _E),
     [("anemia", "low"), ("iron", "low")], None),
    (re.compile(_B + _ANAEMIA + r"\s+(?:due\s+to|secondary\s+to|from|caused\s+by)\s+iron[\s-]+deficiency" + _E),
     [("anemia", "low"), ("iron", "low")], None),
    (re.compile(_B + r"(?:pernicious\s+" + _ANAEMIA + r"|(?:vitamin\s+)?b[\s\-]?12[\s-]+deficiency\s+" + _ANAEMIA
                + r"|" + _ANAEMIA + r"\s+(?:due\s+to|secondary\s+to|from)\s+(?:vitamin\s+)?b[\s\-]?12[\s-]+deficiency)" + _E),
     [("anemia", "low"), ("vitamin_b12", "low")], None),
    (re.compile(_B + r"(?:folate|folic\s+acid)[\s-]+deficiency\s+" + _ANAEMIA + _E),
     [("anemia", "low"), ("folate", "low")], None),
    (re.compile(_B + r"megaloblastic\s+" + _ANAEMIA + _E),
     [("anemia", "low"), ("vitamin_b12", "low"), ("folate", "low")], "uncertain_nutrients"),
    (re.compile(_B + r"(?:sideropeni[ac]|hypoferritin(?:a)?emia|hypoferr?(?:a)?emia|iron\s+(?:depletion|depleted)|ferropeni[ac])" + _E),
     [("iron", "low")], None),
    (re.compile(_B + r"hypovitaminosis\s+d" + _E), [("vitamin_d", "low")], None),
    (re.compile(_B + r"hypovitaminosis\s+b[\s\-]?12" + _E), [("vitamin_b12", "low")], None),
    (re.compile(_B + r"hypocalc(?:a)?emi(?:a|c)" + _E), [("calcium", "low")], None),
    (re.compile(_B + r"hypercalc(?:a)?emi(?:a|c)" + _E), [("calcium", "high")], None),
    (re.compile(_B + r"(?:hypervitaminosis\s+d|vitamin\s+d\s+(?:toxicity|intoxication))" + _E), [("vitamin_d", "high")], None),
    (re.compile(_B + r"(?:iron\s+overload|h(?:a)?emochromatosis)" + _E), [("iron", "high")], None),
    (re.compile(_B + r"an(?:a)?emi(?:a|c)s?" + _E), [("anemia", "low")], None),
]

# "..., now normal" : a statement about the SAME nutrient with its name left out
_ELIDED_NORMAL = re.compile(
    r"(?:,|;|\band\b|\bbut\b)?\s*(?:(?:and|but)\s+)?(?:(?:is|are|has\s+(?:since\s+)?(?:been|become|returned\s+to)|have\s+(?:since\s+)?(?:been|become|returned\s+to)|returned\s+to)\s+)?"
    r"(?:now|currently|today|at\s+present|presently|since\s+then)\s+(?:been\s+|become\s+|returned\s+to\s+)?"
    r"(?:(?:within\s+)?normal(?:\s+(?:range|limits))?|resolved|normali[sz]ed|corrected|replete|adequate|sufficient|in\s+range)" + _E
)

# ---------------------------------------------------------------------------
# Context (assertion) cues
# ---------------------------------------------------------------------------

_W = r"(?<![a-z])"
_WE = r"(?![a-z])"

_NEG_PRE = re.compile(
    _W + r"(?:no|not|without|denies|denied|deny|denying|never|none|nil|negative\s+for|absence\s+of|absent|free\s+of|"
    r"ruled\s+out|rules\s+out|excluded|exclude|(?:do|does|did|is|are|was|were|has|have|had|can|could|would|should)n't)" + _WE
)
_NEG_POST = re.compile(
    _W + r"(?:ruled\s+out|excluded|absent|negative|not\s+(?:been\s+)?(?:confirmed|identified|detected|observed|found|seen|present|evident)|"
    r"was\s+not\s+found|is\s+not\s+present|(?:has|have)\s+been\s+(?:ruled|excluded))" + _WE
)
_DOUBLE_NEG = re.compile(
    _W + r"(?:cannot\s+(?:be\s+)?(?:rule[ds]?|exclude[ds]?)\s*(?:out)?|can\s+not\s+be\s+(?:ruled\s+out|excluded)|can't\s+(?:be\s+)?(?:rule|exclude)\w*(?:\s+out)?|"
    r"not\s+(?:yet\s+)?(?:been\s+)?(?:ruled\s+out|excluded)|unable\s+to\s+(?:rule\s+out|exclude)|without\s+ruling\s+out|"
    r"may\s+not\s+be\s+(?:ruled\s+out|excluded)|cannot\s+be\s+excluded)" + _WE
)
_UNCERTAIN_PRE = re.compile(
    _W + r"(?:suspected|suspicion\s+of|suspect(?:s|ed)?|possible|possibly|probable|probably|likely|may\s+have|might\s+have|may\s+be|might\s+be|"
    r"could\s+be|query|to\s+(?:rule\s+out|exclude|evaluate|investigate|assess|check|confirm)|rule\s+out|r/o|evaluate\s+for|assess(?:ment)?\s+for|"
    r"investigate|investigation\s+of|work[\s-]?up\s+for|screen(?:ing)?\s+for|concern\s+for|concerning\s+for|consistent\s+with|"
    r"suggestive\s+of|suggests?|suggesting|compatible\s+with|differential|ddx|hypothesis|perhaps|maybe|questionable|awaiting|pending|"
    r"under\s+investigation|being\s+investigated|presumed|presumptive|appears?\s+to\s+(?:have|be)|seems?\s+to\s+(?:have|be)|"
    r"signs?\s+of|symptoms?\s+of|suspicious\s+for|worrisome\s+for|consider(?:ing)?|looking\s+for|check\s+for|test\s+for|testing\s+for)" + _WE + r"|\?"
)
_UNCERTAIN_POST = re.compile(
    _W + r"(?:suspected|possible|probable|likely|unlikely|to\s+be\s+(?:confirmed|determined|clarified|investigated|excluded|ruled\s+out)|"
    r"unconfirmed|pending|under\s+investigation|to\s+rule\s+out|to\s+exclude|needs?\s+(?:confirmation|to\s+be\s+confirmed))" + _WE + r"|\?"
)
_HIST_PRE = re.compile(
    _W + r"(?:history\s+of|hx\s+of|hx|h/o|past\s+(?:medical\s+)?history\s+of|past\s+history|prior|previous(?:ly)?|formerly|former|"
    r"in\s+the\s+past|used\s+to\s+have|earlier|remote|prior\s+episode\s+of|antecedents?|(?:was|were)\s+(?:previously\s+)?(?:diagnosed|treated)\s+(?:with|for)|"
    r"treated\s+for|status\s+post|s/p|as\s+a\s+child|in\s+childhood|childhood)" + _WE
)
_HIST_POST = re.compile(
    _W + r"(?:treated|resolved|corrected|normali[sz]ed|reversed|overcome|in\s+the\s+past|previously|as\s+a\s+child|in\s+childhood|"
    r"\d+\s+(?:years?|months?)\s+ago|years?\s+ago|in\s+(?:19|20)\d{2}|since\s+resolved|post[\s-]?treatment|after\s+treatment|has\s+resolved|had\s+resolved)" + _WE
)
_FAMILY_PRE = re.compile(
    _W + r"(?:mother|father|sister|brother|son|daughter|grandmother|grandfather|grandparents?|aunt|uncle|cousin|parents?|siblings?|"
    r"relatives?|family\s+history|familial|maternal|paternal|family\s+members?|husband|wife|spouse|niece|nephew|mom|dad|grandma|grandpa|family)" + _WE
)
_FAMILY_POST = re.compile(
    _W + r"(?:in\s+(?:the\s+)?(?:mother|father|family|sister|brother|relatives?)|on\s+the\s+(?:maternal|paternal)\s+side|"
    r"in\s+(?:a\s+)?(?:first[\s-]degree\s+)?relatives?|familial|family\s+history|runs\s+in\s+the\s+family)" + _WE
)
_HYPO_PRE = re.compile(
    _W + r"(?:risk\s+of|at\s+risk\s+(?:of|for)|prevention\s+of|prevent(?:ing)?|avoid(?:ing)?|in\s+case(?:\s+of)?|if|should|whether|when|unless|"
    r"(?:might|may|can|could|will|would)\s+(?:cause|lead\s+to|result\s+in|provoke|produce|develop)|counsell?ing|education|advice|"
    r"prophylaxis|predispos\w+\s+to|increases?\s+the\s+risk\s+of|risk\s+factors?\s+for|prone\s+to|monitor\s+for|to\s+prevent|to\s+avoid|"
    r"in\s+order\s+to\s+prevent|strategies?\s+(?:for|to\s+prevent))" + _WE
)
_TREATMENT_CTX = re.compile(
    _W + r"(?:under\s+treatment|on\s+treatment|in\s+treatment|being\s+treated|ongoing\s+treatment|currently\s+treated|"
    r"on\s+(?:iron|vitamin|calcium|supplement\w*|replacement)|receiving\s+(?:treatment|supplement\w*|replacement)|supplemented|"
    r"on\s+supplementation|taking\s+supplements?)" + _WE
)
_TERMINATOR = re.compile(
    _W + r"(?:but|however|although|though|except|apart\s+from|aside\s+from|other\s+than|yet|whereas|while|whilst|despite|nevertheless)" + _WE
    + r"|,\s*(?:with|has|have|shows?|showing|presents?|presenting|exhibits?|confirmed|diagnosed|and\s+(?:has|have|shows?|presents?|confirmed|diagnosed))" + _WE
    + r"|" + _W + r"and\s+(?:has|have|shows?|presents?|exhibits?|confirmed|diagnosed|there\s+is|with)" + _WE
)
_SENT_BREAK = re.compile(r"[.;!](?=\s|$)|\n")
_CONJ = re.compile(_W + r"(?:or|and|nor)" + _WE)


@dataclass
class Mention:
    target: str
    polarity: str            # low | normal | high
    stance: str              # affirmed | denied | uncertain | historical | ignored
    assertion: str           # affirmed|negated|uncertain|historical|family|hypothetical
    level: str               # deficient | borderline  (only for polarity low)
    kind: str                # term | descriptor | special | elided
    text: str
    start: int
    end: int
    under_treatment: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class _Candidate:
    start: int
    end: int
    kind: str
    items: list[tuple[str, str]]
    level_hint: str = "deficient"
    forced: str | None = None


class MentionExtractor:
    """Extract explicit mentions + their assertion context."""

    # ------------------------------------------------------------------
    WINDOW = 1200   # a sentence is never examined beyond +-WINDOW characters: keeps every lookup O(1) on huge inputs

    @staticmethod
    def _sentence_bounds(norm: str, pos: int) -> tuple[int, int]:
        w = MentionExtractor.WINDOW
        lo = max(0, pos - w)
        start = lo
        for m in _SENT_BREAK.finditer(norm, lo, pos):
            start = m.end()
        m2 = _SENT_BREAK.search(norm, pos, min(len(norm), pos + w))
        end = m2.start() if m2 else min(len(norm), pos + w)
        return start, end

    def clause_bounds(self, norm: str, s: int, e: int) -> tuple[int, int]:
        """Character span of the CLAUSE around a mention ("No iron deficiency, but low B12" -> 2 clauses).

        Used to give the trained classifier the part of the sentence that actually belongs to the mention.
        """
        sent_s, sent_e = self._sentence_bounds(norm, s)
        sent = norm[sent_s:sent_e]
        rel_s, rel_e = s - sent_s, max(e - sent_s, 0)
        a, b = 0, len(sent)
        for t in _TERMINATOR.finditer(sent):
            if t.end() <= rel_s:
                a = t.end() if not t.group(0).startswith(",") else t.start() + 1
            elif t.start() >= rel_e:
                b = t.start()
                break
        return sent_s + a, sent_s + b

    def _scope(self, norm: str, s: int, e: int, kind: str) -> tuple[str, str]:
        sent_s, sent_e = self._sentence_bounds(norm, s)
        sent = norm[sent_s:sent_e]
        rel_s, rel_e = s - sent_s, max(e - sent_s, 0)
        scope_start, scope_end = 0, len(sent)
        for t in _TERMINATOR.finditer(sent):
            if t.end() <= rel_s:
                scope_start = t.end() if not t.group(0).startswith(",") else t.start() + 1
            elif t.start() >= rel_e:
                scope_end = t.start()
                break
        pre = sent[scope_start:rel_s]
        post = sent[rel_e:scope_end]
        if kind == "descriptor" and "," in pre:
            seg = pre[pre.rfind(",") + 1:]
            if not _CONJ.search(seg):
                pre = seg
        return pre, post

    @staticmethod
    def _within(rx: re.Pattern, text: str, max_words: int, from_end: bool) -> bool:
        ms = list(rx.finditer(text))
        if not ms:
            return False
        if from_end:  # cue BEFORE the mention: distance from cue end to text end
            return any(len(text[m.end():].split()) <= max_words for m in ms)
        return any(len(text[:m.start()].split()) <= max_words for m in ms)  # cue AFTER

    def context(self, norm: str, s: int, e: int, kind: str = "term") -> tuple[str, bool]:
        pre, post = self._scope(norm, s, e, kind)
        treat = bool(_TREATMENT_CTX.search(pre) or _TREATMENT_CTX.search(post))
        if _DOUBLE_NEG.search(pre) or _DOUBLE_NEG.search(post):
            return "uncertain", treat
        if self._within(_NEG_PRE, pre, 12, True) or self._within(_NEG_POST, post, 4, False):
            return "negated", treat
        if self._within(_FAMILY_PRE, pre, 6, True) or self._within(_FAMILY_POST, post, 3, False):
            return "family", treat
        if self._within(_HYPO_PRE, pre, 10, True):
            return "hypothetical", treat
        if self._within(_UNCERTAIN_PRE, pre, 10, True) or self._within(_UNCERTAIN_POST, post, 4, False):
            return "uncertain", treat
        if self._within(_HIST_PRE, pre, 10, True) or self._within(_HIST_POST, post, 6, False):
            return "historical", treat
        return "affirmed", treat

    # ------------------------------------------------------------------
    @staticmethod
    def _targets_in(list_text: str) -> list[str]:
        found = []
        for key, rx in _ALIAS_BY_TARGET.items():
            if rx.search(list_text):
                found.append(key)
        return found

    def _candidates(self, norm: str) -> list[_Candidate]:
        cands: list[_Candidate] = []
        for m in _PATTERN_A.finditer(norm):
            term = m.group("term")
            hint = "borderline" if "insufficien" in term else "deficient"
            items = [(t, "low") for t in self._targets_in(m.group("list")) if t != "anemia"]
            if items:
                cands.append(_Candidate(m.start(), m.end(), "term", items, hint))
        for m in _PATTERN_B.finditer(norm):
            pol = "low" if m.group("low") else "normal" if m.group("normal") else "high"
            hint = "borderline" if (m.group("low") and "insufficien" in m.group("low")) else "deficient"
            items = [(t, pol) for t in self._targets_in(m.group("list"))]
            if items:
                noun_like = bool(m.group("low") and re.fullmatch(r"deficiency|deficient|insufficiency|depletion|deficit|shortage|lacking", m.group("low")))
                cands.append(_Candidate(m.start(), m.end(), "term" if noun_like else "descriptor", items, hint))
        for rx, items, forced in _SPECIAL:
            for m in rx.finditer(norm):
                cands.append(_Candidate(m.start(), m.end(), "special", list(items), "deficient", forced))
        # longest-first greedy, non-overlapping
        cands.sort(key=lambda c: (c.start, -(c.end - c.start)))
        accepted: list[_Candidate] = []
        last_end = -1
        for c in cands:
            if c.start >= last_end:
                accepted.append(c)
                last_end = c.end
        return accepted

    @staticmethod
    def _stance(polarity: str, assertion: str) -> str:
        if polarity == "low":
            return {"affirmed": "affirmed", "negated": "denied", "uncertain": "uncertain",
                    "historical": "historical"}.get(assertion, "ignored")
        if polarity == "normal":
            return {"affirmed": "denied", "negated": "uncertain"}.get(assertion, "ignored")
        # high: not a deficiency
        return "denied" if assertion == "affirmed" else "ignored"

    MAX_MENTIONS = 500

    def extract(self, norm: str, original: str | None = None) -> list[Mention]:
        original = original if original is not None else norm
        mentions: list[Mention] = []
        cands = self._candidates(norm)[: self.MAX_MENTIONS]
        for c in cands:
            assertion, treat = self.context(norm, c.start, c.end, c.kind)
            for target, pol in c.items:
                eff_assertion = assertion
                if c.forced == "uncertain_nutrients" and target != "anemia" and assertion == "affirmed":
                    eff_assertion = "uncertain"
                level = c.level_hint if (pol == "low" and target == "vitamin_d") else "deficient"
                mentions.append(Mention(
                    target=target, polarity=pol,
                    stance=self._stance(pol, eff_assertion), assertion=eff_assertion,
                    level=level, kind=c.kind, text=original[c.start:c.end].strip(),
                    start=c.start, end=c.end, under_treatment=treat,
                ))
        self._elided_normals(norm, original, cands, mentions)
        return mentions

    def _elided_normals(self, norm: str, original: str, cands: list[_Candidate], mentions: list[Mention]) -> None:
        """"History of low B12, now normal": the second clause names no nutrient; it refers to the previous mention."""
        for m in _ELIDED_NORMAL.finditer(norm):
            prev = [c for c in cands if c.end <= m.start() + 2]
            if not prev:
                continue
            c = prev[-1]
            gap = norm[c.end:m.start()]
            if _SENT_BREAK.search(gap) or len(gap.split()) > 6:
                continue
            if any(x.kind == "elided" and x.start == m.start() for x in mentions):
                continue
            lows = [(t, p) for t, p in c.items if p == "low" and t != "anemia"] or [(t, p) for t, p in c.items if p == "low"]
            if not lows:
                continue
            prev_ments = [x for x in mentions if x.start == c.start and x.end == c.end]
            prev_assert = prev_ments[0].assertion if prev_ments else "affirmed"
            if prev_assert in ("negated", "family", "hypothetical"):
                continue
            for x in prev_ments:
                if x.polarity == "low" and x.assertion == "affirmed":      # "was low, now normal": the low value is in the past
                    x.assertion, x.stance = "historical", "historical"
            for t, _p in lows:
                mentions.append(Mention(
                    target=t, polarity="normal", stance=self._stance("normal", "affirmed"), assertion="affirmed",
                    level="deficient", kind="elided", text=original[m.start():m.end()].strip(" ,;"),
                    start=m.start(), end=m.end(),
                ))

    # ------------------------------------------------------------------
    def _hints(self, norm: str, table, keep: tuple[str, ...] = ("affirmed", "historical", "uncertain")) -> list[dict]:
        out = []
        for label, rx_s, targets in table:
            rx = re.compile(_B + "(?:" + rx_s + ")" + _E)
            for m in rx.finditer(norm):
                assertion, _ = self.context(norm, m.start(), m.end(), "term")
                if assertion not in keep:
                    continue
                out.append({"label": label, "targets": targets, "text": m.group(0), "assertion": assertion,
                            "start": m.start(), "end": m.end()})
                break
        return out

    def symptoms(self, norm: str) -> list[dict]:
        return self._hints(norm, SYMPTOMS)

    def symptom_statements(self, norm: str) -> list[dict]:
        """Every symptom mention INCLUDING denied ones (used for cross-report contradiction checks)."""
        return self._hints(norm, SYMPTOMS, keep=("affirmed", "historical", "uncertain", "negated"))

    def risk_factors(self, norm: str) -> list[dict]:
        return self._hints(norm, RISK_FACTORS)

    def treatments(self, norm: str, original: str | None = None, include_negated: bool = False) -> list[dict]:
        original = original if original is not None else norm
        out: list[dict] = []
        seen: set[tuple[str, str]] = set()
        verb_list = re.compile(_B + r"(?:" + SUPPLEMENT_VERBS + r")\s*(?:of|with|on)?\s*(?:(?:the|a|an|any|daily)\s+)?(?P<list>" + _NUTR_LIST + r")")
        found: list[tuple[int, int, list[str]]] = []
        for m in verb_list.finditer(norm):
            found.append((m.start(), m.end(), [t for t in self._targets_in(m.group("list")) if t != "anemia"]))
        for rx_s, targets in TREATMENT_NAMES:
            for m in re.compile(_B + "(?:" + rx_s + ")").finditer(norm):
                found.append((m.start(), m.end(), targets))
        for s, e, targets in found:
            assertion, _ = self.context(norm, s, e, "term")
            if assertion in ("family", "hypothetical") or (assertion == "negated" and not include_negated):
                continue
            for t in targets:
                key = (t, norm[s:e])
                if key in seen:
                    continue
                seen.add(key)
                out.append({"target": t, "text": original[s:e].strip(), "assertion": assertion, "start": s, "end": e})
        return out

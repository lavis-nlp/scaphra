"""
scaphra - a scattered phrase matcher

In some languages such as german, phrases describing, for example, a
problem may be scattered throughout different places inside a sentence.
Most NER systems or pattern matcher assume continous spans of mentions.
This program aims to offer a fast and easy to configure matcher for
scattered phrases. It is implemented as a SpaCy component and thus
relies on its tokenizer, sentencizer, and lemmatizer. Additionally,
a stemming library is used to increase recall.

* An example pattern to match: "A D"
* In a sentence "A B C D E."
* Will produce a match with character positions [(0, 1), (6, 7)]
* As long as the max distance parameter is greater than 3.

"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Callable, Optional

import spacy
from ktz.collections import buckets, unbucket

from lib import cistem

log = logging.getLogger(__name__)


# all this global state fiddling makes me a sad panda :(
spacy.tokens.Token.set_extension("stem", getter=lambda t: cistem.stem(t.text.lower()))


def spanify(lis):
    # [1,2,3,4,6,8,9] -> [(1, 3), (6, 7), (8, 10)]
    # [1, 3, 5] -> [(1,2), (3,4), (5,6)]
    # pretty c-like... is there a more _pythonic_ solution?
    i = 0
    while i < len(lis):
        j = i

        while j + 1 < len(lis) and lis[j + 1] - lis[j] == 1:
            j += 1

        yield lis[i], lis[j] + 1
        i = j + 1


@dataclass(frozen=True)
class Partial:
    """A partial match.

    Partial matches are used to keep track of partially matched
    phrases. They are accepted or rejected based on the constraints
    provided to Scaphra and whether they are fully matched eventually.
    """

    id: int
    positions: tuple[int]

    def describe(self, doc, patterns) -> str:
        pattern = " ".join(patterns[self.id])
        mention = " ".join(map(str, (doc[i] for i in self.positions)))
        return f"Partial of [{pattern}]: matched '{mention}'"


# pre-compiled regexes
# e.g. () ,, results in: (^[^,()]+$|(,.*,|\(.*\)))
#
# which applies:
#   T: a b
#   T: a , b , c
#   T: a ( b ) c
#   F: a , b
#   F: a ( b
#   F: a ) b
def _re_compile_between():
    patterns = "()", ",,"
    escaped = (r"\(", r"\)"), (",", ",")

    # '()', ',,' -> '(),'
    disallowed = "".join(set("".join(patterns)))

    # ( ) , , -> (.*)|,.*,
    groups = "|".join(list(map(lambda t: ".*".join(t), escaped)))

    regex = f"^[^{disallowed}]*$|{groups}"
    return re.compile(regex, re.MULTILINE)


RE_BETWEEN = _re_compile_between()
HOMAG_MATCHER = "homag_matcher"


class Scaphra:
    """Scattered Phrase Matcher.

    TODO long description

    Approach
    --------

    Track matched patterns in dictionaries: Each key is a possible
    match which is compared to the token at hand. The values are
    partial match tuples: (<ID>, <MATCHED>) where <MATCHED> tracks the
    matched tokens.

    Runtime Performance
    -------------------

    Should be a near O(n*m) solution, mostly independent of pattern
    count, where n is the number of texts and m the number of tokens.
    For a detailed description see the comments in Scaphra.match.

    Note regarding stemming
    -----------------------

    It seems as if stemming will never be a part of spacy
    https://github.com/explosion/spaCy/issues/327 however, it fails to
    identify "stumpfen" if "stumpf" is given which is not acceptable.
    Hence Scaphra.match will also regard "externally" stemmed
    phrases (using cistem).

    """

    max_space: Optional[int]  # maximum allowed token distance
    patterns: list[tuple[str]]  # all patterns
    patternmap: dict[tuple[str], int]  # pattern to mention

    def expand_phrasemap(
        self,
        phrasemap: dict[str, list[str]],
    ) -> dict[str, list[str]]:

        if not self.augment:
            return phrasemap

        sourcemap = {key: phrases.copy() for key, phrases in phrasemap.items()}

        for key, phrase in [
            (key, phrase.strip().split())
            for key, phrases in sourcemap.items()
            for phrase in phrases
        ]:

            for augmented in self.augment(phrase):
                # cannot use sets because of spacy
                if augmented not in phrasemap[key]:
                    phrasemap[key].append(augmented)

        log.info(f"expanded to {sum(map(len, phrasemap.values()))} phrases")
        return phrasemap

    def __init__(
        self,
        phrasemap: dict[str, list[str]],
        nlp,
        augment: Callable[[tuple[str]], set[tuple[str]]] = None,
        max_space: Optional[int] = None,
        n_process: Optional[int] = None,
    ):
        self.augment = augment
        self.max_space = max_space

        log.info(f"init matcher from {sum(map(len, phrasemap.values()))} phrases")
        self.phrasemap = self.expand_phrasemap(phrasemap)

        # see match() for intendet purpose
        # while inserting: items must be unique
        self.patterns: set[tuple[str]] = set()
        self.patternmap = defaultdict(set)

        raw = [
            (phrase, key) for key, phrases in phrasemap.items() for phrase in phrases
        ]

        # create key -> doc mapping for later matcher.add
        for doc, key in nlp.pipe(raw, as_tuples=True, n_process=n_process or 1):

            stems = tuple(token._.stem for token in doc)
            lemmata = tuple(token.lemma_ for token in doc)

            for tokens in (stems, lemmata):
                self.patterns.add(tokens)
                self.patternmap[tokens].add(key)

        # when using: items must be indexable
        self.patterns: list[tuple[str]] = list(self.patterns)
        log.info(f"created {len(self.patternmap)} patterns")

    def _match_retain(self, doc, pos: int, part: Partial) -> bool:
        if self.max_space is not None and pos + part.positions[-1] >= self.max_space:
            return False

        if len(part.positions) > 1:
            lower, upper = part.positions[-2:]

            # remove partial matches that exceeded the maximum token span
            if self.max_space is not None and upper - lower >= self.max_space:
                return False

            # remove partial matches that have undesired content in between
            between = str(doc[lower + 1 : upper])
            matches = RE_BETWEEN.search(between)

            if "\n" in between or not matches:
                return False

        return True

    def _match_group(self, doc, matches: set[Partial]):
        groups = {}
        for part in matches:
            keys = self.patternmap[self.patterns[part.id]]

            for key in keys:

                # create unique identifier
                posrep = ".".join(map(str, part.positions))
                name = f"match:{key}:{posrep}"

                spans = []
                for rg in spanify(part.positions):
                    span = spacy.tokens.Span(doc, *rg, label=key)
                    spans.append(span)

                group = spacy.tokens.SpanGroup(doc=doc, name=name, spans=spans)

                # eliminate duplicate matches
                groups[name] = group

        return groups

    def _filter_partials(
        self,
        doc,
        pos: int,
        partials: dict[str, list[Partial]],
        matches: set[Partial],
    ):
        new_part = defaultdict(list)

        # get all matching base patterns and partial matches
        for old_part in partials:

            # add current position to the position aggregator
            positions = old_part.positions + (pos,)

            # create new partial match (updated positions)
            part = replace(old_part, positions=positions)

            # look up original pattern
            pattern = self.patterns[part.id]

            if not self._match_retain(doc=doc, pos=pos, part=part):
                continue

            if len(pattern) == len(part.positions):
                matches.add(part)
                continue

            # advance matched partial match by the next token
            new_part[pattern[len(positions)]].append(part)

        return new_part

    def match(self, doc) -> list[spacy.tokens.Span]:
        def by_first_token(idx, pattern):
            return pattern[0], Partial(id=idx, positions=())

        # keep track of base patterns
        # patterns: [('zu', 'spaet'), ... ]
        # pat_base['zu'] -> [(0, ()), ...]
        pat_base = defaultdict(list, buckets(self.patterns, by_first_token))

        # keep track of partial matches (same structure as pat_base)
        pat_part = defaultdict(list)

        # aggregate matched patterns
        matches: set[Partial] = set()

        # get position lemma, and stem for each token in the document
        gen = ((pos, token._.stem, token.lemma_) for pos, token in enumerate(doc))
        # look at lemmas and stems independently
        gen = ((pos, token) for pos, stem, lemma in gen for token in (stem, lemma))

        # look at every token and aggregate matches (as it is opaque
        # to the caller we don't need to differentiate between lemma
        # and stem matches as long as the mentions are unique in the end)
        for pos, token in gen:

            # MATCH
            # consume all partial and base patterns and either add them to
            # the matches or retain them as partial matches
            joined = pat_base[token] + pat_part[token]
            new_part = self._filter_partials(
                doc,
                pos=pos,
                partials=joined,
                matches=matches,
            )

            # FILTER AND RESCUE
            # former partials are filtered and if they are
            # allowed to stay moved to the new partial matches
            for key, part in unbucket(pat_part):
                retain = self._match_retain(doc=doc, pos=pos, part=part)

                if not retain:
                    continue

                new_part[key].append(part)

            pat_part = new_part

        # create span groups
        groups = self._match_group(doc=doc, matches=matches)
        return groups

    def __call__(self, doc):
        for name, group in self.match(doc).items():
            doc.spans[name] = group

        return doc

    @staticmethod
    @spacy.language.Language.factory("scaphra")
    def create(
        nlp,
        name: str,
        phrasemap: dict[str, list[str]],
        max_space: Optional[int] = None,
        n_process: Optional[int] = None,
    ):
        return Scaphra(
            phrasemap=phrasemap,
            nlp=nlp,
            max_space=max_space,
            n_process=n_process,
        )

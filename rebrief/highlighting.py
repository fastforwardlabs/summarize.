# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import functools
import itertools
import string
from typing import List, Iterator, Optional, Tuple

import attr

MIN_SIZE = 2


@attr.s(frozen=True, eq=False)
class Word(object):
    # How the word appears in the text
    display: str = attr.ib()
    # What whitespace follows
    tail_ws: str = attr.ib()
    # What to match on
    _match_on: str = attr.ib()

    @_match_on.default
    def _match_on_from_display(self):
        result = self.display.lower()
        for p in string.punctuation:
            result = result.replace(p, "")
        return result

    def __eq__(self, other: "Word") -> bool:
        return self._match_on == other._match_on

    def __ne__(self, other: "Word") -> bool:
        return not (self == other)

    def __bool__(self) -> bool:
        return self._match_on != ""


def _concat_words(words: List[Word]) -> str:
    result = list()
    for i, word in enumerate(words):
        result.append(word.display)
        if i < len(words) - 1:
            # More words coming
            result.append(word.tail_ws)
    return "".join(result)


@attr.s(frozen=True)
class Range(object):
    st: int = attr.ib()
    en: int = attr.ib()

    def size(self) -> int:
        return self.en - self.st

    def __bool__(self) -> bool:
        return self.size() != 0

    def __iter__(self) -> Iterator[int]:
        for i in range(self.st, self.en):
            yield i

    def __contains__(self, other: "Range") -> bool:
        return self.st <= other.st < other.en <= self.en

    def all_subranges(self) -> Iterator["Range"]:
        """Of size at least MIN_SIZE."""
        inds = list(range(self.st, self.en+1))
        all_subr = list()
        for st, en in itertools.combinations(inds, 2):
            if st > en:
                st, en = en, st
            all_subr.append(Range(st=st, en=en))
        # Put the biggest first
        all_subr.sort(key=lambda r: -r.size())
        
        for r in all_subr:
            if r.size() < MIN_SIZE:
                break
            yield r


class Document(object):
    """Document should be treated as immutable."""
    def __init__(self, words: List[Word]):
        self._words = words
        self._hash = 0
        for word in words:
            self._hash ^= hash(word.display)
        
    def __hash__(self):
        return self._hash

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._words[key]
        if isinstance(key, Range):
            words = list()
            for i in key:
                words.append(self[i])
            return words

    @property
    def words(self) -> List[Word]:
        return self._words


@attr.s
class HighlightedDocument(object):
    document: Document = attr.ib()
    ranges: List[Range] = attr.ib()

    def __str__(self) -> str:
        result = list()
        for i, rng in enumerate(self.ranges):
            result.append(f"Range {i}")
            result.append(_concat_words(self.document[rng]))
            result.append("")
        return "\n".join(result)

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def from_text(text: str) -> "HighlightedDocument":
        text = text.strip()

        words: List[Word] = list()
        def try_append_word(word: Word) -> None:
            nonlocal words
            if word:
                words.append(word)

        mode = "WORD"
        buffer_display: List[str] = list()
        buffer_ws: List[str] = list()
        for char in text:
            is_ws = char in string.whitespace
            is_word = not is_ws

            if is_word and mode == "WS":
                # Record the word and reset buffers
                try_append_word(Word(display="".join(buffer_display), tail_ws="".join(buffer_ws)))
                buffer_display = list()
                buffer_ws = list()

            if is_word:
                buffer_display.append(char)
                mode = "WORD"
            if is_ws:
                buffer_ws.append(char)
                mode = "WS"
        # One more word at the end
        try_append_word(Word(display="".join(buffer_display), tail_ws="".join(buffer_ws)))

        return HighlightedDocument(
            document=Document(words),
            ranges=[Range(0, len(words))]
        )



@functools.lru_cache(100000)
def span_equal(document_x: Document, range_x: Range, document_y: Document, range_y: Range) -> bool:
    if range_x.size() != range_y.size():
        return False
    for xi, yi in zip(range_x, range_y):
        if document_x[xi] != document_y[yi]:
            return False
    return True


@functools.lru_cache(1000)
def span_in(document_x: Document, range_x: Range, document_y: Document, range_y: Range) -> Optional[Range]:
    """If it finds the range in document_y, return that Range.  None means not found."""
    if range_x.size() > range_y.size():
        return None

    # Look at all subranges
    for target_st in range_y:
        target_en = target_st + range_x.size()
        if target_en > range_y.en:
            # We've gone past the end
            break

        try_range = Range(st=target_st, en=target_en)
        if span_equal(document_x, range_x, document_y, try_range):
            # Found it
            return try_range

    return None


def _find_longest_subspan_in(document_x: Document, range_x: Range, document_y: Document, range_y: Range) -> Optional[Tuple[Range, Range]]:
    """If found, return the longest subspan in x and y."""
    for subrange_x in range_x.all_subranges():
        if subrange_y := span_in(document_x, subrange_x, document_y, range_y):
            return (subrange_x, subrange_y)
    return None

        
def _find_longest_text(summary: HighlightedDocument, original: HighlightedDocument) -> Optional[Tuple[Range, Range]]:
    """Finds longest consecutive segment shared bewteen summary and original."""
    ans = list()
    for sum_range, orig_range in itertools.product(summary.ranges, original.ranges):
        if subranges := _find_longest_subspan_in(summary.document, sum_range, original.document, orig_range):
            ans.append(subranges)
    
    if not ans:
        return None

    # Return longest
    return sorted(ans, key=lambda r: -r[0].size())[0]


def _subtract_ranges(minuend: List[Range], subtrahend: Range) -> List[Range]:
    subtracted = False

    result = list()
    for range in minuend:
        if subtrahend in range:
            subtracted = True
            if left := Range(range.st, subtrahend.st):
                result.append(left)
            if right := Range(subtrahend.en, range.en):
                result.append(right)
        else:
            result.append(range)

    assert subtracted
    return result


def match_most_text(text: str, original_text: str) -> List[str]:
    """ Finds longest consecutive segments in "text" that also occur in "original_text". """
    summary = HighlightedDocument.from_text(text)
    original = HighlightedDocument.from_text(original_text)

    snippets = []
    while ranges := _find_longest_text(summary, original):
        sum_range, orig_range = ranges
        snippets.append(_concat_words([original.document[i] for i in orig_range]))
        summary.ranges = _subtract_ranges(summary.ranges, sum_range)
        original.ranges = _subtract_ranges(original.ranges, orig_range)
    return snippets


def highlight_text(snippets, article):
    """ Add HTML higlighting around each snippet found in the article. """
    for snippet in snippets: 
        try:
            idx = article.index(snippet)
        except ValueError:
            pass
        else:
            new_article = ( 
                article[:idx] 
                + '<span style="background-color: #FFFF00"> **'
                + article[idx : idx+len(snippet)]
                + '** </span>'
                + article[idx+len(snippet):]
            )
            article = new_article
    return article

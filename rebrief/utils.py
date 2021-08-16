from itertools import combinations
from typing import List, Optional


def match_most_text(text: str, original_text: str) -> List[str]:
    """ Finds longest consecutive segments in "text" that also occur in "original_text". """
    snippets = []
    texts = [text]
    originals = [original_text]
    while longest := _find_longest_text(texts, originals):
        # keep the longest
        snippets.append(longest)
        # remove longest from text and original_text
        new_texts = []
        for text in texts:
            new_texts += text.split(longest)
        new_originals = []
        for text in originals:
            new_originals += text.split(longest)
        texts = new_texts 
        originals = new_originals 
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

def _find_longest_text(texts: List[str], original_texts: List[str]) -> Optional[str]:
    """ Finds longest consecutive segment from one of the strings in `texts` compared to 
        each of the strings in `original_texts`. """
    candidates = []
    for text in texts:
        for original in original_texts:
            if chunk := _find_longest_text_single(text, original):
                candidates.append(chunk)
    if candidates:
        return sorted(candidates, key=lambda x: len(x), reverse=True)[0]
        
def _find_longest_text_single(text: str, original_text: str) -> Optional[str]:
    """ Finds longest consecutive segment shared bewteen text and original_text."""
    words = text.split()
    # Generative transformer models have a tendancy to place periods between 
    # sentences with whitespace on each side -- remove these
    words = [word for word in words if word is not "."]
    pairs_of_indices = list(combinations(range(len(words)), 2))
    pairs_of_indices.sort(key= lambda x: x[1]-x[0], reverse=True)
    for pair in pairs_of_indices:
        test_chunk = " ".join(words[pair[0]:pair[1]+1])
        if test_chunk in original_text:
            return test_chunk
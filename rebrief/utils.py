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

import os
import pathlib
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
    words = [word for word in words if word != "."]
    pairs_of_indices = list(combinations(range(len(words)), 2))
    pairs_of_indices.sort(key= lambda x: x[1]-x[0], reverse=True)
    for pair in pairs_of_indices:
        test_chunk = " ".join(words[pair[0]:pair[1]+1])
        if test_chunk in original_text:
            return test_chunk

def create_path(pathname: str) -> None:
    """Creates the directory for the given path if it doesn't already exist."""
    dir = str(pathlib.Path(pathname).parent)
    if not os.path.exists(dir):
        os.makedirs(dir)

def absolute_pathname(*paths) -> str:
    """Given a path relative to this project's top-level directory, returns the
    full path in the OS.
    Args:
        paths: A list of folders/files.  These will be joined in order with "/"
            or "\" depending on platform.
    Returns:
        The full absolute path in the OS.
    """
    # First parent gets the scripts directory, and the second gets the top-level.
    result_path = pathlib.Path(__file__).resolve().parent.parent
    for path in paths:
        result_path /= path
    return str(result_path)
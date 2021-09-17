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
import re
from bs4 import BeautifulSoup
import wikipediaapi as wikiapi

"""
The wikipedia package has broken functionality -- calling page.sections should return 
a list of section titles but instead returns an empty list.

The wikipedia-api package can access section titles but it has to be done recursively,
which is confusing and unnecessary. On the plus side, this package also does a lot of 
pre-parsing of the original wiki HTML making it easier to manage. 

So both packages are used to extract section titles and their respective text! 
It ain't pretty but it works. 
"""

WIKIAPI = wikiapi.Wikipedia(language="en", extract_format=wikiapi.ExtractFormat.HTML)
EXCLUDE = [
    "Journals",
    "Conferences",
    "See also",
    "References",
    "Further reading",
    "External links",
]


def extract_headings(article_title):
    """Extract the main headings and subheadings from a Wikipedia page.

    Only returns top and second level headings (deeper subsections are ignored).
    Returned headings are indented according to their hierarchical level for
    display in the Streamlit dropdown box.
    """
    p_html = WIKIAPI.page(article_title).text
    headings = []
    soup = BeautifulSoup(p_html, features="html.parser")
    for header in soup.find_all(re.compile("h[2-3]")):
        if header.text not in EXCLUDE:
            prefix = "-- " if header.name == "h3" else ""
            headings.append(prefix + header.text)
    return headings


def cleanup(txt: str) -> str:
    """Contains a function to cleanup text following wiki parsing.

    Covered cases are shown in the test file.
    """
    supported_punctuation = ".,!?"
    for c in supported_punctuation:
        txt = txt.replace(f" {c} ", f"{c} ")
    # Put a little space in.
    return re.sub(r"([" + supported_punctuation + r"])([A-Z])", r"\1 \2", txt)

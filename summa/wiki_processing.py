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

WIKIAPI = wikiapi.Wikipedia(language='en', extract_format=wikiapi.ExtractFormat.HTML)
EXCLUDE = ['Journals', 'Conferences', 'See also', 'References', "Further reading", 'External links']

def extract_headings(article_title):
    """ Extract the main headings and subheadings from a Wikipedia page. 

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
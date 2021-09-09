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

from selenium import webdriver  
from selenium.webdriver.chrome.options import Options
import wikipedia as wiki
import streamlit as st
st.set_page_config(layout="wide")

from summa.wiki_parsing import cleanup, extract_headings
from summa.highlighting import match_most_text, highlight_text
from summa.st_model_wrappers import (
    abstractive, 
    modern_extractive, 
    classic_extractive, 
    hybrid_extractive
)

CHROMEDRIVER_PATH = '/Users/mbeck/Projects/chromedriver'
SCREENSHOT_WINDOW_SIZE = "1500,900" 

MODELS = (
    abstractive,
    modern_extractive,
    classic_extractive,
    hybrid_extractive
)

# Global versions for caching
@st.cache(allow_output_mutation=True)
def load_model(model):
    return model.load()

# Global versions for caching
@st.cache(allow_output_mutation=True)
def summarize_text(article, model):
    return model.summarize(article, load_model(model))

def make_url(text, url):
    """ Add HTML to convert text into a clickable url. """
    new_text = f'<a target="_blank" href="{url}">{text}</a>'
    return new_text

# ============ SIDEBAR ============
st.sidebar.image("images/fflogo1@1x.png")
st.sidebar.markdown("**Summarize.** is a text summarization prototype that demonstrates \
    _abstractive_ and _extractive_ summarization, and showcases Transformer-based models \
    as well as classic and hybrid models. Read more on our [blog.](TODO: LINK).")

# ----- Model Selection -----
model_selector = {m.display_name: m for m in MODELS}
model_obj = model_selector[
    st.sidebar.selectbox("Choose a summarization model.", list(model_selector.keys()))
    ]
st.sidebar.markdown("#### Model description")
st.sidebar.markdown(model_obj.description, unsafe_allow_html=True)

# ============ MAIN PAGE ============

# ----- Display ToC and Text Box -----
top_left, top_right = st.beta_columns(2)
top_left.title("Summarize.")

# ----- Article Selection -----
articles = {
    "Machine learning": wiki.search("machine learning", results=1),
    "Birds": wiki.search("birds", results=1),
    "Knitting": wiki.search("knitting", results=1),
    "Baking": wiki.search("baking", results=1),
    "Jeopardy!": wiki.search("jeopardy", results=1),
} 
article_selection = top_left.selectbox("Choose a Wikipedia ariticle.", list(articles.keys()))

# ----- Load Stuff -----
wiki_page = wiki.page(article_selection, auto_suggest=False)
headings = ["Front Matter"] + extract_headings(article_selection)
top_left.subheader("Table of Contents")
section_selection = top_left.selectbox("Choose a subsection.", headings)

if "--" in section_selection: 
    section_selection = section_selection.split("-- ")[1]
if section_selection == "Front Matter":
    article = wiki_page.summary 
else:
    article = wiki_page.section(section_selection).strip()

text = top_right.text_area("Subection text -- or enter your own text to summarize!", article, height=350)
text = cleanup(text)

# ----- Summarize Text -----
all_text = summarize_text(text, model_obj)

# ----- Display Highlighting & Results -----
bottom_left, bottom_right = st.beta_columns(2)
bottom_left.markdown(f"## Original text: {section_selection} \n (Highlighted segments denote model's summary.)")

# ----- Highlighting -----
for s in all_text.texts:
    if not s.summary:
        break
    snippets = match_most_text(s.summary, s.text)
    s.text = highlight_text(snippets, s.text)

for paragraph in str(all_text).split("\n"):
    bottom_left.write(paragraph, unsafe_allow_html=True)

# ----- Results -----
bottom_right.markdown(f"## {model_obj.display_name} Summary")
bottom_right.write(f"\n{all_text.summary()}")


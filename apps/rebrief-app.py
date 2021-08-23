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

import re
from dataclasses import dataclass

from selenium import webdriver  
from selenium.webdriver.chrome.options import Options
import wikipedia as wiki
import streamlit as st
st.set_page_config(layout="wide")

from rebrief.utils import match_most_text, highlight_text
from rebrief.st_model_wrappers import (
    abstractive, 
    modern_extractive, 
    classic_extractive, 
    upgraded_classic_extractive
)

CHROMEDRIVER_PATH = '/Users/mbeck/Projects/chromedriver'
SCREENSHOT_WINDOW_SIZE = "1500,900" 

MODELS = (
    abstractive,
    modern_extractive,
    classic_extractive,
    upgraded_classic_extractive
)

# Global versions for caching
@st.cache(allow_output_mutation=True)
def load_model(model):
    return model.load()

# Global versions for caching
@st.cache(allow_output_mutation=True)
def summarize_text(article, model):
    return model.summarize(article, load_model(model))

def make_screenshot(url, output_filename):
    """ Use a headless browser to take a screenshot of the provided url and save to file. """
    chrome_options = Options()  
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--window-size=%s" % SCREENSHOT_WINDOW_SIZE)
    if not url.startswith('http'):
        raise Exception('URLs need to start with "http"')

    driver = webdriver.Chrome(
        executable_path=CHROMEDRIVER_PATH,
        options=chrome_options
    )  
    driver.get(url)
    driver.save_screenshot(output_filename)
    driver.close()
    return

def make_url(text, url):
    """ Add HTML to convert text into a clickable url. """
    new_text = f'<a target="_blank" href="{url}">{text}</a>'
    return new_text


# ============ SIDEBAR ============
st.sidebar.title("Summarization on Wikipedia articles")
st.sidebar.markdown("Currently using HuggingFace models for Abstractive Summarization. But as you'll see below, \
    the summaries are often verbatim from the original text (highlighted).")

# ----- Article Selection -----
articles = {
    "Machine learning": wiki.search("machine learning", results=1),
    "Birds": wiki.search("birds", results=1),
    "Knitting": wiki.search("knitting", results=1),
    "Baking": wiki.search("baking", results=1),
    "Jeopardy!": wiki.search("jeopardy", results=1),
} 
st.sidebar.markdown("### Select Things.")
article_selection = st.sidebar.selectbox("Choose one of my hobbies to summarize", list(articles.keys()))

# ----- Model Selection -----
model_selector = {m.display_name: m for m in MODELS}
model_obj = model_selector[
    st.sidebar.selectbox("Choose a summarization model:", list(model_selector.keys()))
    ]

# ----- Model Description -----
st.sidebar.markdown(model_obj.description)

# ============ MAIN PAGE ============
# load article
wiki_page = wiki.page(article_selection, auto_suggest=False)
article = wiki_page.summary

# ----- Display Article Title and Image -----
title_url = make_url(article_selection, wiki_page.url)
image_name = f"images/{article_selection.replace(' ', '_')}.png"
screenshot = make_screenshot(wiki_page.url, image_name)

_, img_col, _ = st.beta_columns(3)
img_col.markdown(f"### {title_url}", unsafe_allow_html=True)
img_col.image(image_name)

# ----- Display Summary and original -----
col1, col2 = st.beta_columns(2)

# extract answers
summary = summarize_text(article, model_obj)

col1.subheader('Model Summary')
col1.write(f"\n{summary}")

col2.subheader('Original Wikipedia article')
col2.markdown("##### (actually, only the first section which is itself a summary)")

snippets = match_most_text(summary, article)
highlighted_article = highlight_text(snippets, article)
for paragraph in highlighted_article.split("\n"):
    col2.write(paragraph, unsafe_allow_html=True)


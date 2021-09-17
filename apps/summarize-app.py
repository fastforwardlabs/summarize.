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

import matplotlib.pyplot as plt
import pandas as pd
import wikipedia as wiki
import streamlit as st
st.set_page_config(layout="wide")

from summa.wiki_parsing import extract_headings, cleanup
from summa.highlighting import match_most_text, highlight_text

from st_model_wrappers import (
    abstractive, 
    modern_extractive, 
    classic_extractive, 
    hybrid_extractive
)

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

@st.cache(allow_output_mutation=True)
def load_data(selection):
    if selection == "wiki":
        return pd.read_csv("data/wikipedia/ml_excerpts.pd")
    elif selection == "cnn":
        return pd.read_csv("data/cnn_dailymail/sAMPle.pd")

def make_url(text, url):
    """ Add HTML to convert text into a clickable url. """
    new_text = f'<a target="_blank" href="{url}">{text}</a>'
    return new_text

def make_bar_chart(df, idx, model_name):
    labels = ['Neural Abstractive', 'Neural Extractive', 'Classic Extractive', 'Hybrid Extractive']
    values = df[labels].iloc[idx].values
    to_highlight = labels.index(model_name)

    fig, ax = plt.subplots()
    ax.bar([0,1,2,3], height = values)
    ax.bar([to_highlight], height = values[to_highlight], color="orange")
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('ROUGE-L score')
    return fig

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
top_left, top_right = st.columns(2)
top_left.title("Summarize.")

# ----- Text Selection -----
text_options = {
    'Excerpts from Wikipedia page on Machine Learning': "wiki",
    'Articles from the CNN/Daily News dataset': "cnn",
}
text_selection = top_left.selectbox("Choose a passage to summarize.", list(text_options.keys()))
text_selection = text_options[text_selection]

# ----- Load Stuff -----
if text_selection == "wiki":
    df = load_data(text_selection)
    df.fillna('', inplace=True)
    text_options = {r.heading: i for i, r in df.iterrows()}
    top_left.subheader("Table of Contents")
    selection = top_left.selectbox("Choose a subsection.", list(text_options.keys()))
else:
    df = load_data(text_selection)
    text_options = {r.title: i for i, r in df.iterrows()}
    selection = top_left.selectbox("Choose a news article.", list(text_options.keys()))  

row_idx = text_options[selection]
article = df.iloc[row_idx]['article']

original = cleanup(article)
text = top_right.text_area("Subection text -- or enter your own text to summarize!", article, height=300)
text = cleanup(text)

if text == original:
    summary = df.iloc[row_idx][model_obj.name+"_summary"]
else:
    # ----- Summarize Text -----
    summary = summarize_text(text, model_obj)

# ----- Display Highlighting & Results -----
bottom_left, bottom_right = st.columns(2)
bottom_left.markdown(f"## Original text (summary highlighted)")

# ----- Highlighting -----
snippets = []
if summary:
    snippets = match_most_text(summary, text)
if snippets:
    highlighted_article = highlight_text(snippets, text)
    for paragraph in highlighted_article.split("\n"):
        bottom_left.write(paragraph, unsafe_allow_html=True)
else:
    bottom_left.write(text)

# ----- Results -----
bottom_right.markdown(f"## {model_obj.display_name} Summary")
bottom_right.write(f"\n{summary}")

if text == original:
    if text_selection == "cnn":
        with bottom_right.expander("Qualitative Comparison"):
            st.write("Because the CNN/Daily Mail dataset includes gold standard summaries, \
                we can do a quantitative comparison with our model output. The standard approach \
                is to compute the ROUGE score between the model's output and the gold standard.")
            st.pyplot(make_bar_chart(df, row_idx, model_obj.display_name))
            st.write("There are several flavors of ROUGE score. ROUGE-L considers the longest common subsequence in the summary. ")
    if text_selection == "wiki":
        with bottom_right.expander("Qualitative Comparison"):
            st.markdown(df.iloc[row_idx]['commentary'])

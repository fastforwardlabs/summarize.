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

import attr
import transformers as trf

from rebrief.models.classic_extractive import (
    SentenceTextRank, 
    build_classic_nlp_pipeline,
    build_trf_nlp_pipeline, 
    classic_summary, 
    sentence_summary_upgrade
)
from rebrief.models.neural_extractive import (
    SentenceBertClass, 
    load_neural_extractive_model,
    summarize
)


@attr.s()
class SummarizationModel(object):
    """
    Wrapper class for a summarization model for use in the ReBrief Streamlit app.

    This wrapper class is designed to abstract away the complexities of juggling
    multiple different summarization models in the ReBrief Streamlit app.  
    The load and summarize methods should point to functions that will be called during 
    execution of the ReBrief app. These functions are designed to operate independently
    so that loading and summarizing functionality can be cached separately in the 
    Streamlit app. This is also why this class does not store the actual model object, 
    once loaded. 

    name        (str) short, general name for the model
    load        (uncalled) function that loads the model (from HF Model repo, spaCy, etc.).
                This function accepts no arguments and returns the model.

                Example:
                def load_HF_model():
                    return trf.pipeline("summarization") # this loads a full summarization pipeline

    summarize   (uncalled) function that generates a summary from a longer document.
                This function must accept a document (str), and a SummarizationModel object
                and returns a summary (str)

                Example:
                def summary(document, model):
                    # assuming the HF Summarization pipeline is passed as the model
                    return model(document)[0]['summary_text'] 

    display_name    (str) longer/more detailed description for display in the model selection 
                    box of the ReBrief Streamlit app
    description     (str) Description of the model for display under the model selection box 
                    in the ReBrief Streamlit app

    __hash__    For Streamlit cacheing 
    """
    name:str = attr.ib()
    load = attr.ib()
    summarize = attr.ib()
    display_name = attr.ib()
    description:str = attr.ib()

    def __hash__(self):
        return self.name


def load_abstractive_model():
    return trf.pipeline("summarization")

def abstractive_summary(text, model):
    try:
        output = model(text, return_tensors=False, clean_up_tokenization_spaces=True)
        summary = output[0]['summary_text']
    except IndexError:
        # the input text is too long. Need to break it up. 
        paragraphs = text.split("\n")
        paragraphs = [p for p in paragraphs if p]
        summary = []
        for paragraph in paragraphs:
            try:
                output = model(paragraph, return_tensors=False, clean_up_tokenization_spaces=True)
                summary.append(output[0]['summary_text'])
            except IndexError:
                # if a paragraph is STILL too long, split further
                sentences = paragraph.split(".") 
                # TODO: need to generalize this because these chunks might be too long
                chunks = 2 
                segment_size = int(len(sentences)/chunks)
                while sentences:
                    segment = ". ".join(sentences[:segment_size])
                    sentences = sentences[segment_size:]
                    output = model(segment, return_tensors=False, clean_up_tokenization_spaces=True)
                    summary.append(output[0]['summary_text'])
        summary = "\n".join(summary)
    return summary

abstractive = SummarizationModel(
    name = "abstractive",
    load = load_abstractive_model,
    summarize = abstractive_summary,
    display_name = "Neural Abstractive",
    description = """### HF Summarization Pipeline \n
    I don't know what model this is. Pegasus? Or T5? Or is Pegasus a variant of T5?
    # TODO: read about this! 
    """, 
)

modern_extractive = SummarizationModel(
    name = "modern_extractive",
    load = load_neural_extractive_model,
    summarize = summarize,
    display_name = "Neural Extractive",
    description = """### SentenceBERT fine-tuned for summarization\n 
    
    """,
)

classic_extractive = SummarizationModel(
    name = "classic_extractive",
    load = build_classic_nlp_pipeline,
    summarize = classic_summary,
    display_name = "Classic Extractive",
    description = "### TextRank \n TextRank is a graph-based ranking algorithm, \
    which provides a way of determining the importance of a vertex within a graph, \
    given global information drawn recursively from the entire graph.\
    \n The basic idea behind textrank is that of 'voting' -- \
    when one vertex is linked to another, it's essentially voting for that other vertext. \
    The more votes a vertext has, the more important it is. Additionally, a vertex's importance \
    influences how important its vote is!  So a vertex's score is determined not only by the \
    number of votes it receives from other vertices, but also their importance scores. \
    \n While the vertexes can represent anything, in this classic version they are \
    instantiated by the words found in the text (after removing stop words). \
    The edges between the vertices (the 'votes') are initialized as the co-occurrence \
    between two words within a given context window size. After initialization, \
    the PageRank algorithm (of search engine fame) computes the recursive scoring. \
    These scores are used to determine the most important words and phrases in the text \
    and sentences containing the top phrases are extracted as a summary.",
)

upgraded_classic_extractive = SummarizationModel(
    name = "hybrid_extractive",
    load = build_trf_nlp_pipeline,
    summarize = sentence_summary_upgrade,
    display_name = "Hybrid Extractive",
    description = """ ### TextRank + SentenceBERT\n
    This hybrid approach uses the basic tenents of the Classic Extractive model but with a twist.
    \n
    We still use TextRank to build a graph. But instead of words, 
    each vertex represents a full sentence in the document. 
    The edges are initialized as the cosine similarity between two sentence
    representations after processing through SentenceBERT.  
    Once initalized, the PageRank algorithm is run to determine the final 
    score of each sentence in the document and the sentences with the highest
    scores are selected as the document summary.
    """,
)
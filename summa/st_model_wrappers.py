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

from summa.models import neural_extractive as ne
from summa.models import neural_abstractive as na
from summa.models import classic_extractive as ce

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

abstractive = SummarizationModel(
    name = "abstractive",
    load = na.load_abstractive_model,
    summarize = na.abstractive_summary,
    display_name = "Neural Abstractive",
    description = "HuggingFace Summarization Pipeline.\n\n HuggingFace \
    provide models that perform _abstractive_ summarization. \
    These models ingest a document and generate text word by word \
    (or token by token) until a summary of a desired length is achieved. \
    \n\n While these models currently represent the state-of-the-art in text \
    summarization, they have some drawbacks. Namely, as Transformers, these\
    models are limited in the amount of text they can process at one time. \
    Secondly, Transformers are typically more computationally demanding than \
    traditional models. And finally, any model that generates text word by word \
    can occasionally produce inaccurate or factually incorrect summaries. \
    \n\n This HF summarization pipeline loads a distilBART model -- a \"distilled\" \
    version of Facebook's BART model -- which is 25% more computationally efficient \
    than the original while matching the larger model's accuracy. It is trained\
    on the CNN/Daily Mail dataset, a standard for summarization tasks.", 
)

modern_extractive = SummarizationModel(
    name = "modern_extractive",
    load = ne.load_neural_extractive_model,
    summarize = ne.summarize,
    display_name = "Neural Extractive",
    description = "Fine-tuning SentenceBERT.\n\n For this model \
    we train a Transformer to perform _extractive_ rather than _abstractive_ summarization. \
    While the details of the approach can be found in our blog post, [Extractive \
    Summarization with SentenceBERT](TODO: LINK), here's the gist. \
    \n\n The CNN/Daily Mail dataset includes news articles as well as human-generated\
    \"highlights\", which provide an article summary. We identify the sentences in the \
    article that most closely match those from the highlights and assign them the label \
    \"In Summary\"; all other sentences from the article are assigned a label of \"Not in Summary\". \
    For each news article in the training set we compute a full article representation, \
    as well as individual sentence representations with SentenceBERT. \
    We pass these representations to a dense layer and train/fine-tune SentenceBERT using the \
    binary labels we extracted. This results in a model that, at inference time, computes \
    a score for each sentence in a document. Those with the highest scores are extracted as \
    the document summary.",
)

classic_extractive = SummarizationModel(
    name = "classic_extractive",
    load = ce.build_classic_nlp_pipeline,
    summarize = ce.classic_summary,
    display_name = "Classic Extractive",
    description = "TextRank.\n\n TextRank is a classic graph-based ranking \
    algorithm that computes the importance of a vertex given global information \
    about the entire graph. \
    \n\n The basic idea is that of \"voting\": when one vertex is linked to another, \
    it's essentially casting a vote for that other vertex. The more votes a vertex \
    has, the more important it is. Additionally, a vote from an important vertex \
    counts for more than one from a less important vertex. Ultimately, a vertex's \
    score is determined not only by the number of votes it receives \
    but also by the importances of the vertices casting the votes. \
    \n\n While the vertices in the graph can represent anything, in this classic \
    version each vertex represents a word from the document (after removing stop words). \
    The edges between the vertices (the \"votes\") are initialized as the co-occurrence \
    between those words within a given context window size. After initialization, \
    the PageRank algorithm (of search engine fame) computes the recursive scoring. \
    These scores are used to determine the most important words and phrases in the document, \
    and sentences containing the top phrases are extracted as a summary.",
)

hybrid_extractive = SummarizationModel(
    name = "hybrid_extractive",
    load = ce.build_trf_nlp_pipeline,
    summarize = ce.sentence_summary_trf,
    display_name = "Hybrid Extractive",
    description = "TextRank + SentenceBERT.\n\n This hybrid approach relies on the \
    same basic tenets of the \"Classic Extractive\" model but with a twist.\
    \n\n We still use TextRank to build a graph, but now each vertex represents a sentence \
    from the document, rather than a single word. A numerical representation of each sentence is \
    computed via the SentenceBERT Transformer model. The edges of the graph are then initialized \
    as the cosine similarity between each pair of sentence representations. \
    The PageRank algorithm computes the final importance scores for each sentence in the \
    document. Sentences with the highest scores are selected as the document summary.",
)
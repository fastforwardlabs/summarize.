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

import typing

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.tokens import Doc 
import pytextrank
import torch
import numpy as np

# better sentencebert model
SBERT_MODEL_NAME = 'sentence-transformers/paraphrase-mpnet-base-v2'
SBERT_CONFIG = {
    "model": {
        "@architectures": "spacy-transformers.TransformerModel.v1",
        "name": SBERT_MODEL_NAME,
        "get_spans": {
            "@span_getters": "spacy-transformers.sent_spans.v1"
        }
    }
}

def build_classic_nlp_pipeline():
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank", config={ "token_lookback": 10 })
    return nlp

def build_trf_nlp_pipeline(config=SBERT_CONFIG):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("sentencizer")
    transformer = nlp.add_pipe("transformer", config=SBERT_CONFIG)
    # initialize the transformer model
    try:
        transformer.model.initialize([nlp.make_doc("Jello world I am here. Hello world.")])
    except ValueError:
        transformer.model.initialize([nlp.make_doc("Jello world I am here. Hello world.")])
    return nlp


class SentenceTextRank:
    def __init__(self, doc): 
        self.doc = doc
        self.transformer_ranks = self.trfembeddings_textrank()
        self.wordembedding_ranks = self.wordembeddings_textrank()
        self.sentences = [sent for sent in doc.sents]

    def _sentence_rank(self, sentence_vectors):
        """ Rank sentences by importance.
        
        Takes a list of sentence_vectors and computes pair-wise cosine similarity
        between all sentences. A graph is constructed with the sentence_vectors as 
        nodes and cosine similarity as edges. 
        Pagerank algorithm is run on the graph and scores returned.
        """
        cossims = cosine_similarity(sentence_vectors)
        # cosine similarity occasionally returns negative values
        # but pagerank doesn't work on negative values, so make them not negative.
        cossims[cossims < 0] = 0
        nx_graph = nx.from_numpy_array(cossims)
        return nx.pagerank(nx_graph)

    def trfembeddings_textrank(self):
        sentence_vectors = self.get_transformer_embeddings()
        return self._sentence_rank(np.array(sentence_vectors.cpu()))  

    def wordembeddings_textrank(self):
        sentence_vectors = self.get_sentence_embeddings()
        return self._sentence_rank(sentence_vectors)

    def get_transformer_embeddings(self):
        # This does take attention into account and it makes a big difference!
        token_embeddings = torch.tensor(self.doc._.trf_data.tensors[0])
        attn_mask = self.doc._.trf_data.tokens['attention_mask']
        return self._mean_pooling(token_embeddings, attn_mask)

    def get_sentence_embeddings(self):
        try:
          # if using spaCy on GPU, convert cupy arr --> numpy arr with get()
          return [sent.vector.get() for sent in self.doc.sents] 
        except:
          return [sent.vector for sent in self.doc.sents] 

    def _mean_pooling(self, token_embeddings, attention_mask):
        """ Take attention mask into account for correct averaging of sentence-bert tokens """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_summary(self, *, transformer_ranks=False, limit_sentences=5, preserve_order=True, return_scores=False):
        """ Generate an extractive summary of the doc. """
        if transformer_ranks:
            scores = self.transformer_ranks
        else:
            scores = self.wordembedding_ranks
        # order by rank
        ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(self.sentences)), reverse=True)
        summary = ranked_sentences[:limit_sentences]
        if preserve_order: 
            # rerank by index in the doc
            summary = sorted(summary, key=lambda x: x[1])
        if return_scores:
            return [(s, str(sent)) for s, i, sent in summary]
        else:
            return " ".join([str(sent) for s, i, sent in summary])
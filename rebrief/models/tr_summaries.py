import typing
from spacy import Language 
from rebrief.models import SentenceTextRank

NUM_SENTENCES = 3

def classic_summary(text:str, nlp:Language, **kwargs) -> str:
    """ Generate summary with classic TextRank model. 

    TextRank model uses PageRank algorithm on word co-ocurrences to 
    determine important phrases. Sentences containing the highest ranked
    phrases are extracted as the document summary. 
    """
    doc = nlp(text)
    tr = doc._.textrank
    summary = []
    for sentence in tr.summary(
                        limit_phrases=10, 
                        limit_sentences=NUM_SENTENCES, 
                        preserve_order=True):
        summary.append(str(sentence))
    return " ".join(summary)

def sentence_summary(text:str, nlp:Language, **kwargs) -> str:
    """ Generate a summary with a TextRank model constructed from sentences.

    This version of TextRank uses the PageRank algorithm on sentence embeddings
    (constructed from an average of spaCy word embeddings) and cosine similarities 
    to determine the most important sentences in the document. The highest ranked
    sentences are extracted as the document summary. 
    """
    try: 
        NUM_SENTENCES = kwargs.pop('num_sentences')
    except:
        pass
    doc = nlp(text)
    sent_tr = SentenceTextRank(doc)
    return sent_tr.generate_summary(transformer_ranks=False, limit_sentences=NUM_SENTENCES)

def sentence_summary_upgrade(text:str, nlp:Language, **kwargs) -> str:
    """ Generate a summary with a TextRank model constructed from sentences.

    This version of TextRank uses the PageRank algorithm on sentence embeddings
    (constructed SentenceBERT embeddings) and cosine similarities to 
    determine the most important sentences in the document. The highest ranked
    sentences are extracted as the document summary. 
    """
    try: 
        NUM_SENTENCES = kwargs.pop('num_sentences')
    except:
        pass
    doc = nlp(text)
    sent_tr = SentenceTextRank(doc)
    return sent_tr.generate_summary(transformer_ranks=True, limit_sentences=NUM_SENTENCES)

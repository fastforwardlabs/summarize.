
import os 
from typing import List, Dict 
import numpy as np
import spacy 
import torch
from transformers import AutoModel, AutoTokenizer

from rebrief.utils import absolute_pathname

BASE_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
MODEL_PATH = absolute_pathname("models/minilm_bal_exsum.pth")
DEVICE = device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_SENTENCES = 3           
BATCH_SIZE = 3              
MIN_SENTENCE_LENGTH = 14    

# I don't know where to put these.... 
nlp = spacy.load('en_core_web_sm') # change to large at some point
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# get mean pooling for sentence bert models 
# ref https://www.sbert.net/examples/applications/computing-embeddings/README.html#sentence-embeddings-with-transformers
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class SentenceBertClass(torch.nn.Module):
    def __init__(self, model_name=BASE_MODEL_NAME):
        super(SentenceBertClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(384*3, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)
        self.classifierSigmoid = torch.nn.Sigmoid()

    def forward(self, sent_ids, doc_ids, sent_mask, doc_mask):

        sent_output = self.l1(input_ids=sent_ids, attention_mask=sent_mask) 
        sentence_embeddings = mean_pooling(sent_output, sent_mask) 

        doc_output = self.l1(input_ids=doc_ids, attention_mask=doc_mask) 
        doc_embeddings = mean_pooling(doc_output, doc_mask)

        # elementwise product of sentence embs and doc embs
        combined_features = sentence_embeddings * doc_embeddings  

        # get concat of both features and elementwise product
        feat_cat = torch.cat((sentence_embeddings, doc_embeddings, combined_features), dim=1)  
        
        pooler = self.pre_classifier(feat_cat) 
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.classifierSigmoid(output) 

        return output

def load_neural_extractive_model(model_path=MODEL_PATH, devide=DEVICE):
    model = SentenceBertClass() 
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    return model

def get_model_inputs(texts, tokenizer):
    return tokenizer.batch_encode_plus(
        texts, 
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )  

def batch_predict(sentence_inputs, document_inputs, model):
    sent_token_ids = sentence_inputs['input_ids']
    sent_attn_mask = sentence_inputs['attention_mask']
    batch_size = len(sent_token_ids)
    
    doc_token_ids  = document_inputs['input_ids'].expand(batch_size, -1) 
    doc_attn_mask = document_inputs['attention_mask'].expand(batch_size, -1)
    
    return model(sent_token_ids, doc_token_ids, sent_attn_mask, doc_attn_mask)

def summarize(text, model, limit_sentences=NUM_SENTENCES, batch_size=BATCH_SIZE, return_scores=False):
    doc = nlp(text)
    doc_inputs = get_model_inputs([text], tokenizer)
    doc_sentences = [str(sent) for sent in doc.sents if len(sent) > MIN_SENTENCE_LENGTH]

    scores = []
    for i in range(int(len(doc_sentences) / batch_size) + 1):
        sentence_batch = doc_sentences[i*batch_size: (i+1) * batch_size]        
        sentence_inputs = get_model_inputs(sentence_batch, tokenizer)
        preds = batch_predict(sentence_inputs, doc_inputs, model)
        scores = scores + preds.tolist()

    sent_pred_list = [{"sentence": doc_sentences[i], "score": scores[i][0], "index":i} for i in range(len(doc_sentences))]
    sorted_sentences = sorted(sent_pred_list, key=lambda k: k['score'], reverse=True) 

    sorted_result = sorted_sentences[:NUM_SENTENCES] 
    sorted_result = sorted(sorted_result, key=lambda k: k['index']) 

    summary = [ x["sentence"] for x in sorted_result]
    summary = " ".join(summary)

    if return_scores:
        return summary, scores
    return summary
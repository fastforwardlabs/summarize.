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

import transformers as trf

# this is the current default when loading the HF summarization pipeline
# making it explicit here to reduce ambiguity.
ABSUM_MODEL = "sshleifer/distilbart-cnn-12-6" 

def load_abstractive_model(model_name = ABSUM_MODEL):
    return trf.pipeline("summarization", model=model_name, tokenizer=model_name)

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

import fasttext
import spacy
import sparknlp
from sparknlp.pretrained import *
from sparknlp.annotator import *
from sparknlp.base import *
# import pandas as pd

import os
import sys

# import time

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

class spacy_lm:
    def __init__(self, weight='en_core_web_lrg'):
        try:
            if spacy.prefer_gpu():
                print('Activate GPU acceleration successfully')
            else:
                print('Fail to activate GPU, using CPU instead...')
        except:
            print('Fail to activate GPU, using CPU instead...')
            
        try:
            self.model = spacy.load(weight)
        except OSError:
            print('Can\'t find model, please run command "python -m spacy download en_core_web_trf" to download it and restart the program')

        

class LM_fasttext:
    '''language model given by fasttext, basically used for word embedding and further enr extraction'''
    def __init__(self, weight_path='utils/cc.en.300.bin'):
        self.model = fasttext.load_model(weight_path)
        
    def get_word_vector(self, word:str):
        return self.model.get_word_vector(word)

    def get_sentence_vector(self, sentence:str):
        return self.model.get_sentence_vector(sentence)
    
class LM_pos:
    '''  language model for pos tagging'''
    def __init__(self):
        self.pipeline = PretrainedPipeline('explain_document_dl', lang = 'en')

    def tagging(self, sentence:str, pos_wanted = ['NNP', 'VB', 'VBN', 'NNS']):
        '''given pos tag you wanted, return words from the sentence in form of list'''
        annotations = self.pipeline.fullAnnotate(sentence)[0]
        return [i.metadata['word'] for i in annotations['pos'] if i.result in pos_wanted]
    
    def full_tagging(self, sentence):
        '''return a dictionary with keys ['word', 'pos']'''
        annotations = self.pipeline.fullAnnotate(sentence)[0]
        word_list = [annotator.metadata['word'] for annotator in annotations['pos']]
        pos_list = [annotator.result for annotator in annotations['pos']]
        return {'word':word_list, 'pos':pos_list}
        
        
class LM_ner:
    def __init__(self):
        ''' language model for NER '''
        self.pipeline = PretrainedPipeline('onto_recognize_entities_bert_base', lang = 'en')
        
    def extraction(self, sentence:str):
        ''' given sentence, directly return entities recognized '''
        annotations = self.pipeline.annotate(sentence)
        if annotations['entities']:
            return annotations['entities']
        else:
            return None

if __name__ == '__main__':
    pos = LM_pos()
    tmp = pos.full_tagging('Given that I like The Lion King, Pocahontas, and The Beauty and the Beast, can you recommend some movies?')
    print(tmp)
    
    
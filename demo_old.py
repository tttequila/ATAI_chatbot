from speakeasypy import Speakeasy
import time
import os
from collections import Counter

import string
import csv
import numpy as np
from sklearn.metrics import pairwise_distances
import spacy
# from spacy.lang.en import English
import fasttext
import rdflib
import nltk
from nltk.corpus import brown

# import re
# import editdistance
# import pickle

# import cv2
# import logging

USER_NAME = 'broil-grandioso-pie_bot'
PWD = 'NrBl_CRF6101Dg'
HOST = 'https://speakeasy.ifi.uzh.ch'

class KG_handler:
    def __init__(self):
        '''
        ### Notices 
            - all2lbl has more "P" item than ent2id dose (ent2lbl:298, ent2id:47)
            - all2lbl contain all entities and relations, while ent2id lacks some relations
            - labels of entity are not unique, but entity URIs are unique
            - labels of relations are unique (at least unique till now)     
            - to obtain query, we need movie name (entity label), relation label and relation URI
        '''
        
        # language model
        print('initializing language model...')
        self.LM = fasttext.load_model('utils/cc.en.300.bin')
        try:
            # self.spacy_model = en_core_web_trf.load()
            self.spacy_model = spacy.load('en_core_web_trf')
        except OSError:
            print('Can\'t find model, please run command "python -m spacy download en_core_web_trf" to download it and restart the program')

        
        
        
        # RDF config
        print('initialize RDF graph...')
        self.graph = rdflib.Graph().parse('utils/14_graph.nt', format='turtle')
        self.WD = rdflib.Namespace('http://www.wikidata.org/entity/')
        self.WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
        self.DDIS = rdflib.Namespace('http://ddis.ch/atai/')
        self.RDFS = rdflib.namespace.RDFS
        self.SCHEMA = rdflib.Namespace('http://schema.org/')      
        
          
        # label relevant
        print('initialize important variables...')
        self.all2lbl = {ent: str(lbl) for ent, lbl in self.graph.subject_objects(self.RDFS.label)}
        self.rel2lbl = {k:v for k, v in self.all2lbl.items() if self._is_rel(k)}
        self.lbl2rel = {lbl: ent for ent, lbl in self.rel2lbl.items()}
        self.rel_lbl_set = set([v for v in self.rel2lbl.values()])
        self.ent2lbl = {k:v for k, v in self.all2lbl.items() if self._is_ent(k)}
        self.lbl2ent = {lbl: ent for ent, lbl in self.ent2lbl.items()}  # not unique
        self.ent_lbl_set = set([v for v in self.ent2lbl.values()])
        # with open('utils/ent_lbl2vec.pkl', 'rb') as f:
        #     self.ent_lbl2vec = pickle.load(f)
        # with open('utils/rel_lbl2vec.pkl', 'rb') as f:
        #     self.rel_lbl2vec = pickle.load(f)
        self.ent_lbl2vec = {k:self.LM.get_word_vector(k) for k in self.ent_lbl_set}
        self.rel_lbl2vec = {k:self.LM.get_word_vector(k) for k in self.rel_lbl_set}
        # self.all_lbl_set
        
        # embedding relevant
        with open('utils/relation_ids.del', 'r') as ifile:
            self.rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        with open('utils/entity_ids.del', 'r') as ifile:
            self.ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.ent_emb = np.load('utils/entity_embeds.npy')
        self.rel_emb = np.load('utils/relation_embeds.npy')
        
        self.synonyms_dict = {
            'cast member' :['actor', 'actress', 'cast'],
            'genre': ['type', 'kind'],
            'publication date': ['release', 'date', 'airdate', 'publication', 'launch', 'broadcast','released','launched'],
            'executive producer': ['showrunner'],
            'screenwriter': ['scriptwriter', 'screenplay', 'teleplay', 'writer', 'script', 'scenarist', 'story'],
            'director of photography': ['cinematographer', 'DOP', 'dop'],
            'film editor': ['editor'],
            'production designer': ['designer'],
            'box office': ['box', 'office', 'funding'],
            'cost': ['budget', 'cost'],
            'nominated for': ['nomination', 'award', 'finalist', 'shortlist', 'selection'],
            'costume designer': ['costume'],
            'official website': ['website', 'site'],
            'filming location': ['flocation'],
            'narrative website': ['nlocation'],
            'production company': ['company'],
            'country of origin': ['origin', 'country'],
            '–' : ['-']
        }
        self.replacement_dict = {k:v for k,  v_list in self.synonyms_dict.items() for v in v_list }
        
    def _ruler_based(self, query:str):
        '''
        Param:
            query: natrual language query from the user
        Return:
            {'ent_lbl': None | str, 'rel_lbl': None | str, 'rel_postfix': None | str}:
            diction of extraction results. if not found than set as None
        '''
        
        res = {'ent_lbl': None,
               'rel': None,
               'rel_lbl': None,
               'rel_postfix': None}
        
        # pre-proecssing
        tokens = self._replace(query)
        
        # get all word combination that match existed entities
        word_seq = [' '.join(tokens[i:j+1]) for i in range(len(tokens)) for j in range(i, len(tokens))]
        matched_seq = [seq for seq in word_seq if seq in (self.ent_lbl_set | self.rel_lbl_set)]      # all exited entities that appear in the sentence
        
        # extraction 
        ent_candidates = []
        for seq in matched_seq:
            if seq in self.ent_lbl_set:
                ent_candidates.append(seq)
            # detected relation
            if seq in self.rel_lbl_set:
                # which means there are two possible word for relations
                if res['rel_lbl'] != None:
                    print("WARNIND: multiple possible relations detected...")
                res['rel'] = seq
                res['rel_lbl'] = self.lbl2rel[seq]
                res['rel_postfix'] = self._get_rel_label(res['rel_lbl'])
        # process possible entities
        ent_candidates = sorted(ent_candidates, key = lambda x:len(x[0]), reverse=True)
        res['ent_lbl'] = ent_candidates[0]
    
        return res
    
    def _similarity_based(self, query, top_k_rel=10, top_k_ent=1):
        '''
        similar to _rule_based(), but return top k similar results in list form, which is:
        Return:
            {'ent_lbl': list(str), 'rel_lbl': list(str), 'rel_postfix': list(str)}:
        '''
        
        res = {'ent_lbl': None,
               'rel': None,
               'rel_lbl': None,
               'rel_postfix': None}
        
        # pre-processing
        tokens = self._replace(query).join()
        
        # token process
        doc = self.spacy_model(tokens)
        
        
        ent = []    # entity
        rel = []    # relation
        for token in doc:
            # print(token)
            if (token.ent_iob_ != "O"):
                ent.append(token.lemma_)
            elif (token.pos_=='NOUN') | (token.pos_=="VERB"):
                rel.append(token.lemma_)
                
                
        # find the closest relation information
        rel_wv = np.array([i for i in self.rel_lbl2vec.values()])
        rel_lbl = [i for i in self.rel_lbl2vec.keys()]
        rel = " ".join(rel)
        wv = self.LM.get_word_vector(rel).reshape((1,-1))
        dist = pairwise_distances(wv, rel_wv).flatten()
        closest_rel_idx = dist.argsort()[:top_k_rel]
        closest_rel_lbl = [rel_lbl[i] for i in closest_rel_idx]
        closest_rel_uri = [self.lbl2rel[i] for i in closest_rel_lbl]
        
        
        # find the closest ent information
        extracted_ent = " ".join(ent)
        ent_wv = np.array([i for i in self.ent_lbl2vec.values()])
        ent_lbl = [i for i in self.ent_lbl2vec.keys()]
        wv = self.LM.get_word_vector(extracted_ent).reshape((1,-1))
        dist = pairwise_distances(wv, ent_wv).flatten()
        closest_ent_idx = dist.argsort()[:top_k_ent]
        closest_ent_lbl = [ent_lbl[i] for i in closest_ent_idx]

        res['rel'] = rel
        res['ent_lbl'] = closest_ent_lbl[0]
        res['rel_lbl'] = closest_rel_lbl
        res['rel_postfix'] = [self._get_rel_label(uri) for uri in closest_rel_uri]

        return res
    
    
    def _replace(self, sent:str) -> list :
        '''
        replace words in sentence if they have relevant replacement in the dictionary
        '''
        
        # remove all possible punctuation in the end of sentence
        cleaned_sent = sent.rstrip(string.punctuation + ' ')
        tokens = cleaned_sent.split()
        tokens = [ self.replacement_dict[token] if token in self.replacement_dict.keys() else token for token in tokens ]
        
        return tokens
        
    def _get_rel_label(self, URI):
        return str(URI).split('/')[-1]
    
    def _is_rel(self, URI):
        label = self._get_rel_label(URI)
        return label[0] == 'P'
    
    def _is_ent(self, URI):
        label = self._get_rel_label(URI)
        return label[0] == 'Q'   
    
    def get_query_res(self, user_input:str) -> tuple(dict, str):
        
        '''
        ### Arg:
            - user_input (str): user input in natural language
            
        ### Return:
            - extraction (dict): extracted tokens consisting of 
            
                'ent_lbl' : entity label
                
                'rel' : orginial relation
                
                'rel_lbl' : pre-defined relation label
                
                'rel_postfix' : pre-defined relation postfix
                   
            - res (str): query result
        '''
        
        extraction = self._ruler_based(user_input)
        # print(extraction)

        # calling backup extraction strategy
        if None in extraction.values():
            back_extraction = self._similarity_based(user_input)
            # replace None item with similarity based extraction
            for k,v in extraction.items():
                if v == None:
                    extraction[k] = back_extraction[k]    

        # print(back_extraction)

        # grab result from graph
        for i in range(len(extraction['rel_postfix'])):
            movie_name = extraction['ent_lbl']
            target_label = extraction['rel_postfix'][i]
            target_name = extraction['rel_lbl'][i]
            if "date" in target_name:
                query = f'''PREFIX ddis: <http://ddis.ch/atai/>

                PREFIX wd: <http://www.wikidata.org/entity/>

                PREFIX wdt: <http://www.wikidata.org/prop/direct/>

                PREFIX schema: <http://schema.org/>

                SELECT ?date WHERE {{
                    ?movie rdfs:label "{movie_name}"@en.

                    ?movie wdt:{target_label} ?date

                }} LIMIT 1'''
            else:
                query = f'''PREFIX ddis: <http://ddis.ch/atai/>

                PREFIX wd: <http://www.wikidata.org/entity/>

                PREFIX wdt: <http://www.wikidata.org/prop/direct/>

                PREFIX schema: <http://schema.org/>

                SELECT ?lbl WHERE {{
                    ?sub rdfs:label "{movie_name}"@en.

                    ?sub wdt:{target_label} ?obj.

                    ?obj rdfs:label ?lbl.

                }} LIMIT 1'''
            query  = query.strip()
            # print(query)
            res = []
            for row, in self.graph.query(query):
                res.append(str(row))
            # print(res)
            
            
            # if no query return
            if len(res) != 0:
                extraction['rel_lbl'] = [extraction['rel_lbl'][i]]
                extraction['rel_postfix'] = [extraction['rel_postfix'][i]]
                res = res[0]
                break
        
        # if both the similarity based and the ruler based fail to extract entity&relation 
        if len(res) == 0:
            ent_id = self.ent2id[self.lbl2ent[extraction['ent_lbl']]]
            rel_id = self.rel2id[self.lbl2rel[extraction['rel_lbl'][0]]]
            
            head = self.ent_emb[ent_id]
            pred = self.rel_emb[rel_id]
            
            lhs = (head + pred).reshape((1, -1))
            
            # select closest entity
            dist = pairwise_distances(lhs, self.ent_emb).flatten()
            most_likely_idx = dist.argsort()[0]
            res = self.ent2lbl[self.id2ent[most_likely_idx]]
            
        return extraction, res    
    
    
    
class Agent:
    def __init__(self, user_name=USER_NAME, pwd=PWD, host=HOST, log_path = 'logs'):
        self.user_name = user_name
        self.pwd = pwd
        self.host = host
        self.activated_room = []
        self.KG_handler = KG_handler()
        # self.log_path = log_path
        
        # login agent
        self.chat_agent = Speakeasy(host=self.host, username=self.user_name, password=self.pwd)
        self.chat_agent.login() 
        # print(os.getcwd())
        self.log_path =  os.getcwd() + '\\' + log_path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
            print("Creat log path as %s"%self.log_path)
        self.log_path = self.log_path + '\\' + '%s.log'%time.strftime("%m-%d_%H_%M_%S", time.localtime())
        # logging.basicConfig(filename='test.log',
        #                     filemode='w', 
        #                     format='%(asctime)s - %(levelname)s - %(message)s',
        #                     datefmt="%d-%M-%Y %H:%M:%S", 
        #                     level=logging.INFO)
        # logging.info("Log start!")
        
        
        nltk.download("universal_tagset")
        nltk.download("brown")

        verb_prep_pairs = [(v[0].lower(), p[0].lower()) 
                         for (v, p) in nltk.bigrams(brown.tagged_words(tagset="universal")) 
                         if v[1] == "VERB" and p[1] == "ADP"]
        self.verb_prep_counts = Counter(verb_prep_pairs)
        self.little_tiny_tagging = spacy.load("en_core_web_sm")
        
        
    def __get_best_preposition(self, verb):
        '''
        given verb, return the proper prep
        '''
        verb = verb.lower()
        prepositions = [(prep, count) for (v, prep), count in self.verb_prep_counts.items() if v == verb]
        if prepositions:
            best_preposition = max(prepositions, key=lambda x: x[1])
            return best_preposition[0]
        else:
            return None

    def __get_word_pos(self, word) -> str:
        '''
        given a word, return pos
        '''
        doc = self.little_tiny_tagging(word) 
        pos_tag = doc[0].pos_  
        return pos_tag

    def __return_ans(self, movie_name, relation, res) -> str:
        '''
        construct answer with movie name, relation and query result
        '''
        if self.__get_word_pos(relation) == "NOUN":
            ans = "The " + relation + " of " + movie_name + " is " + res + "."
        else:
            ans = movie_name + " was " + relation + " " + self.__get_best_preposition(relation) + " " + res + "."
        return ans
    
        
    def __query(self, query:str) -> str:
        extraction, res = self.KG_handler.get_query_res(query)
        return self.__return_ans(movie_name=extraction['ent_lbl'],
                                 relation=extraction['rel'],
                                 res=res)
        
    
    def __real_time_logging(self, msg:str):
        time_format = time.strftime("[%Y-%m-%d %H:%M:%S]\n", time.localtime())
        with open(self.log_path, mode='a') as f:
            f.write(time_format)
            f.write(msg)
        
    def start(self):
        print('start')
        while True:
            rooms = self.chat_agent.get_rooms(active=True)
            # print(1)
            for room in rooms:
                # check whether there is new room 
                room_id = room.room_id

                if room.room_id not in self.activated_room:
                    # logging
                    self.activated_room.append(room_id)
                    # logging.info("New chat room started: %s"%room_id)
                    self.__real_time_logging("New chat room started: %s"%room_id)
                
                # print(room)
                # Retrieve messages from this chat room.
                for message in room.get_messages(only_partner=True, only_new=True):
                    
                    # logging receiving message
                    # logging.info("From room %s\n Received message: %s"%(room_id, message.message))
                    self.__real_time_logging("From room %s\n Received message %s"%(room_id, message.message))
                    # print("msg: ", message.message)
                    try:
                        ans = self.__query(str(message.message))     
                    except:
                        ans = 'Null, query fail or error happened'
                        
                    # print('ans: ', ans) 
                
                    
                    room.post_messages(f"{ans}")
                    # logging.info("To room %s\n Reply message: %s"%(room_id, ans))
                    self.__real_time_logging("To room %s\n Reply message \{%s\}"%(room_id, ans))

                    room.mark_as_processed(message)
                    
                    
                # Retrieve reactions from this chat room.
                for reaction in room.get_reactions(only_new=True):
                    # Implement your agent here #
                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)
                    
if __name__ == '__main__':
    bot = Agent()
    bot.start()                    
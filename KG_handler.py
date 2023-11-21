import string
import csv
import numpy as np
from sklearn.metrics import pairwise_distances
import rdflib
import pickle
from LM import *

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
        # print('initializing language model...')
        # self.LM = fasttext.load_model('utils/cc.en.300.bin')
        # self.spacy_model = spacy_model

        
        
        
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
        with open('utils/ent_lbl2vec.pkl', 'rb') as f:
            self.ent_lbl2vec = pickle.load(f)
        with open('utils/rel_lbl2vec.pkl', 'rb') as f:
            self.rel_lbl2vec = pickle.load(f)
        # self.ent_lbl2vec = {k:self.LM.get_word_vector(k) for k in self.ent_lbl_set}
        # self.rel_lbl2vec = {k:self.LM.get_word_vector(k) for k in self.rel_lbl_set}
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
        self.replacement_dict = {v:k for k,  v_list in self.synonyms_dict.items() for v in v_list }
    
    def _get_entity_name(self,ent_candidates): # Find the movie entity name by select the candidates with Upper letter and largest length.
        for element in ent_candidates:
            if any(c[0].isupper() for c in element):
                print(element)
                return element
        return None

    # def _ruler_based(self, query:str):
    #     '''
    #     Param:
    #         query: natrual language query from the user
    #     Return:
    #         {'ent_lbl': None | str, 'rel_lbl': None | str, 'rel_postfix': None | str}:
    #         diction of extraction results. if not found than set as None
    #     '''
        
    #     res = {'ent_lbl': None,
    #            'rel': None,
    #            'rel_lbl': None,
    #            'rel_postfix': None}
        
    #     # pre-proecssing
    #     tokens = self._replace(query)

    #     word_seq = [' '.join(tokens[i:j+1]) for i in range(len(tokens)) for j in range(i, len(tokens))]
    #     matched_seq = [seq for seq in word_seq if seq in (self.ent_lbl_set | self.rel_lbl_set)]      # all exited entities that appear in the sentence

    #     # extraction 
    #     ent_candidates = []
    #     for seq in matched_seq:
    #         if seq in self.ent_lbl_set:
    #             ent_candidates.append(seq)
    #         # detected relation
    #         if seq in self.rel_lbl_set:
    #             # which means there are two possible word for relations
    #             if res['rel_lbl'] != None:
    #                 print("WARNIND: multiple possible relations detected...")
    #             res['rel'] = seq
    #             res['rel_lbl'] = [seq]
    #             res['rel_postfix'] = [self._get_rel_label(self.lbl2rel[seq])]
                
                
    #     # extract entity from candidates
    #     ent_candidates = sorted(ent_candidates, key = lambda x:len(x), reverse=True)
    #     # res['ent_lbl'] = ent_candidates[0]
    #     res['ent_lbl'] = self._get_entity_name(ent_candidates)

    #     print("Result:")
    #     print(res)
    #     return res
        
    # def _similarity_based(self, query, top_k_rel=10, top_k_ent=1):
    #     '''
    #     similar to _rule_based(), but return top k similar results in list form, which is:
    #     Return:
    #         {'ent_lbl': list(str), 'rel_lbl': list(str), 'rel_postfix': list(str)}:
    #     '''
        
    #     res = {'ent_lbl': None,
    #            'rel': None,
    #            'rel_lbl': None,
    #            'rel_postfix': None}
        
    #     # pre-processing
    #     tokens = ' '.join(self._replace(query))
        
    #     # token process
    #     doc = self.spacy_model(tokens)
        
        
    #     ent = []    # entity
    #     rel = []    # relation
    #     for token in doc:
    #         # print(token)
    #         if (token.ent_iob_ != "O"):
    #             ent.append(token.lemma_)
    #         elif (token.pos_=='NOUN') | (token.pos_=="VERB"):
    #             rel.append(token.lemma_)
                
    #     # fail to extraction entity or relation 
    #     if (len(ent)==0) | (len(rel)==0):
    #             return False
                
    #     # find the closest relation information
    #     rel_wv = np.array([i for i in self.rel_lbl2vec.values()])
    #     rel_lbl = [i for i in self.rel_lbl2vec.keys()]
    #     rel = " ".join(rel)
    #     wv = self.LM.get_word_vector(rel).reshape((1,-1))
    #     dist = pairwise_distances(wv, rel_wv).flatten()
    #     closest_rel_idx = dist.argsort()[:top_k_rel]
    #     closest_rel_lbl = [rel_lbl[i] for i in closest_rel_idx]
    #     closest_rel_uri = [self.lbl2rel[i] for i in closest_rel_lbl]
        
        
    #     # find the closest ent information
    #     extracted_ent = " ".join(ent)
    #     ent_wv = np.array([i for i in self.ent_lbl2vec.values()])
    #     ent_lbl = [i for i in self.ent_lbl2vec.keys()]
    #     wv = self.LM.get_word_vector(extracted_ent).reshape((1,-1))
    #     dist = pairwise_distances(wv, ent_wv).flatten()
    #     closest_ent_idx = dist.argsort()[:top_k_ent]
    #     closest_ent_lbl = [ent_lbl[i] for i in closest_ent_idx]

    #     res['rel'] = rel
    #     res['ent_lbl'] = closest_ent_lbl[0]
    #     res['rel_lbl'] = closest_rel_lbl
    #     res['rel_postfix'] = [self._get_rel_label(uri) for uri in closest_rel_uri]

    #     return res
    
    
    def _replace(self, sent:str) -> list :
        '''
        replace words in sentence if they have relevant replacement in the dictionary
        '''
        
        # remove all possible punctuation in the end of sentence
        if '-' in sent:
            sent = sent.replace('-','–')

        translator = str.maketrans("", "", string.punctuation.replace(":", ""))
        cleaned_sent = sent.translate(translator)
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
    
    # def get_query_res(self, user_input:str) -> tuple:
        
    #     '''
    #     ### Arg:
    #         - user_input (str): user input in natural language
            
    #     ### Return:
    #         - extraction (dict): extracted tokens consisting of 
            
    #             'ent_lbl' : entity label
                
    #             'rel' : orginial relation
                
    #             'rel_lbl' : pre-defined relation label
                
    #             'rel_postfix' : pre-defined relation postfix
                   
    #         - res (str): query result
    #     '''
    #     extraction = self._ruler_based(user_input)
    #     # print(extraction)

    #     # calling backup extraction strategy
    #     if None in extraction.values():
    #         backup_extraction = self._similarity_based(user_input)
    #         # fail to understand sentence
    #         if backup_extraction == False:
    #              return None, False
    #         # replace None item with similarity based extraction
    #         for k,v in extraction.items():
    #             if v == None:
    #                 extraction[k] = backup_extraction[k]    

    #     # print(backup_extraction)
    #     print("******")
    #     print(extraction)

    #     # grab result from graph
    #     for i in range(len(extraction['rel_postfix'])):
    #         movie_name = extraction['ent_lbl']
    #         target_label = extraction['rel_postfix'][i]
    #         target_name = extraction['rel_lbl'][i]
    #         if "date" in target_name:
    #             query = f'''PREFIX ddis: <http://ddis.ch/atai/>

    #             PREFIX wd: <http://www.wikidata.org/entity/>

    #             PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    #             PREFIX schema: <http://schema.org/>

    #             SELECT ?date WHERE {{
    #                 ?movie rdfs:label "{movie_name}"@en.

    #                 ?movie wdt:{target_label} ?date

    #             }} LIMIT 1'''
    #         else:
    #             query = f'''PREFIX ddis: <http://ddis.ch/atai/>

    #             PREFIX wd: <http://www.wikidata.org/entity/>

    #             PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    #             PREFIX schema: <http://schema.org/>

    #             SELECT ?lbl WHERE {{
    #                 ?sub rdfs:label "{movie_name}"@en.

    #                 ?sub wdt:{target_label} ?obj.

    #                 ?obj rdfs:label ?lbl.

    #             }} LIMIT 1'''
    #         query  = query.strip()
    #         # print(query)
    #         res = []
    #         for row, in self.graph.query(query):
    #             res.append(str(row))
    #         # print(res)
            
            
    #         # if no query return
    #         if len(res) != 0:
    #             extraction['rel_lbl'] = [extraction['rel_lbl'][i]]
    #             extraction['rel_postfix'] = [extraction['rel_postfix'][i]]
    #             res = res[0]
    #             break
        
    #     # if both the similarity based and the ruler based fail to extract entity&relation 
    #     if len(res) == 0:
    #         ent_id = self.ent2id[self.lbl2ent[extraction['ent_lbl']]]
    #         rel_id = self.rel2id[self.lbl2rel[extraction['rel_lbl'][0]]]
            
    #         head = self.ent_emb[ent_id]
    #         pred = self.rel_emb[rel_id]
            
    #         lhs = (head + pred).reshape((1, -1))
            
    #         # select closest entity
    #         dist = pairwise_distances(lhs, self.ent_emb).flatten()
    #         most_likely_idx = dist.argsort()[0]
    #         res = self.ent2lbl[self.id2ent[most_likely_idx]]
            
    #     return extraction, res    
    

from KG_handler import *
from LM import *
import pandas as pd
import nltk
from collections import Counter
from nltk.corpus import brown


class fact_querier():
    def __init__(self, KG:KG_handler, LM_pos:LM_pos, LM_ner:LM_ner):
        
        self.KG = KG
        self.pos = LM_pos
        self.ner = LM_ner
                
        self.fasttext = LM_fasttext()
        
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
            'director': ['directed', 'directs'],
            '–' : ['-']
        }
        self.replacement_dict = {v:k for k,  v_list in self.synonyms_dict.items() for v in v_list }
        
        self.crowdsource_data = pd.read_csv('utils/crowd_data.tsv', sep='\t')
        self.subject_list = self.crowdsource_data['Input1ID'].unique()
        self.predicate_list = self.crowdsource_data['Input2ID'].unique()
        
        verb_prep_pairs = [(v[0].lower(), p[0].lower()) 
                    for (v, p) in nltk.bigrams(brown.tagged_words(tagset="universal")) 
                    if v[1] == "VERB" and p[1] == "ADP"]
        self.verb_prep_counts = Counter(verb_prep_pairs)


   
    def _get_label(self, URI):
        return str(URI).split('/')[-1]
    
    def _is_rel(self, URI):
        label = self._get_label(URI)
        return label[0] == 'P'
    
    def _is_ent(self, URI):
        label = self._get_label(URI)
        return label[0] == 'Q'   
    
    def _is_human(self, ent):
        query = '''
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>


        SELECT ?instance
        WHERE {
        
        SERVICE <https://query.wikidata.org/sparql>{    
        
        ?ent rdfs:label "%s"@en.
        ?ent wdt:P31 ?instance.
        
        }
        }'''%(ent)
        
        res = []
        for row in self.KG.graph.query(query):
            # print(row)
            res+=[self._get_label(str(i)) for i in row]
        
        return ('Q5' in res) 
        
    
    
    def _replace(self, sent:str) -> list :
        '''
        replace words in sentence if they have relevant replacement in the dictionary
        '''
        
        # remove all possible punctuation in the end of sentence
        cleaned_sent = sent.rstrip(string.punctuation + ' ')
        tokens = cleaned_sent.split()
        tokens = [ self.replacement_dict[token] if token in self.replacement_dict.keys() else token for token in tokens ]
        
        return tokens
        
            
    # def _rule_based(self, sentence:str):
    #     '''
    #     Param:
    #         sentence: natrual language query from the user
    #     Return:
    #         {'ent_lbl': None | str, 'rel_lbl': None | str, 'rel_postfix': None | str}:
    #         diction of extraction results. if not found than set as None
    #     '''
        
    #     res = {'ent_lbl': None,
    #            'rel_lbl': None}
        
    #     # pre-proecssing
    #     tokens = self._replace(sentence)
        
    #     window_size = 20
    #     word_seq = []
    #     for i in range(len(tokens)):
    #         end = min(len(tokens), i+window_size)
    #         word_seq.append(' '.join(tokens[i:end]))
    #     matched_seq = [seq for seq in word_seq if seq in (self.KG.ent_lbl_set | self.KG.rel_lbl_set)]      # all exited entities that appear in the sentence
        
        
    #     # extraction 
    #     ent_candidates = []
    #     for seq in matched_seq:
    #         if seq in self.KG.ent_lbl_set:
    #             ent_candidates.append(seq)
    #         # detected relation
    #         if seq in self.KG.rel_lbl_set:
    #             # which means there are two possible word for relations
    #             if res['rel_lbl'] != None:
    #                 print("WARNIND: multiple possible relations detected...")
    #             res['rel_lbl'] = seq
                
    #     # process possible entities
    #     ent_candidates = sorted(ent_candidates, key = lambda x:len(x[0]), reverse=True)
    #     res['ent_lbl'] = ent_candidates[0]
    
    #     return res
    
        
    def _rule_based(self, query:str):
        '''
        Param:
            query: natrual language query from the user
        Return:
            {'ent_lbl': None | str, 'rel_lbl': None | str, 'rel_postfix': None | str}:
            diction of extraction results. if not found than set as None
        '''
        
        res = {'ent_lbl': None,
               'rel_lbl': None,
               'rel_text':None}
        
        # pre-proecssing
        tokens = self._replace(query)
        
        # get all word combination that match existed entities
        word_seq = [' '.join(tokens[i:j+1]) for i in range(len(tokens)) for j in range(i, len(tokens))]
        matched_seq = sorted([seq for seq in word_seq if seq in (self.KG.ent_lbl_set | self.KG.rel_lbl_set)], reverse=True)     # all exited entities that appear in the sentence
        
        # extraction 
        ent_candidates = []
        for seq in matched_seq:
            if seq in self.KG.ent_lbl_set:
                ent_candidates.append(seq)
            # detected relation
            if seq in self.KG.rel_lbl_set:
                # which means there are two possible word for relations
                if res['rel_lbl'] != None:
                    print("WARNIND: multiple possible relations detected...")
                res['rel_lbl'] = seq
                res['rel_text'] = seq
        # process possible entities
        ent_candidates = sorted(ent_candidates, key = lambda x:len(x[0]), reverse=True)
        res['ent_lbl'] = ent_candidates[0]
    
        return res
    
    def _model_based(self, query, similarity):
        '''
        similar to _rule_based(), but return top k similar results in list form, which is:
        Return:
            {'ent_lbl': list(str), 'rel_lbl': list(str), 'rel_postfix': list(str)}:
        '''
        
        res = {'ent_lbl': None,
               'rel_lbl': None,
               'rel_text':None}
        
        # pre-processing
        tokens = " ".join(self._replace(query))
        print("*********************", tokens)
        # token process
        rel = self.pos.tagging(tokens, ['NNP', 'VB', 'VBN', 'NNS'])
        ent = self.ner.extraction(tokens)
        print("*********************" ,rel, ent)
        print("*********************" ,ent)
        
        
        
        
        # ent = []    # entity
        # rel = []    # relation
        # for token in doc:
        #     # print(token)
        #     if (token.ent_iob_ != "O"):
        #         ent.append(token.lemma_)
        #     elif (token.pos_=='NOUN') | (token.pos_=="VERB"):
        #         rel.append(token.lemma_)
                
        if similarity:
            # find the closest relation information
            rel_wv = np.array([i for i in self.KG.rel_lbl2vec.values()])
            rel_lbl = [i for i in self.KG.rel_lbl2vec.keys()]
            rel = " ".join(rel)
            wv = self.fasttext.get_word_vector(rel).reshape((1,-1))
            dist = pairwise_distances(wv, rel_wv).flatten()
            closest_rel_idx = dist.argsort()[0]
            closest_rel_lbl = rel_lbl[closest_rel_idx]
            # closest_rel_uri = [self.KG.lbl2rel[i] for i in closest_rel_lbl]
            
            
            # find the closest ent information
            extracted_ent = ent[0]
            ent_wv = np.array([i for i in self.KG.ent_lbl2vec.values()])
            ent_lbl = [i for i in self.KG.ent_lbl2vec.keys()]
            wv = self.fasttext.get_word_vector(extracted_ent).reshape((1,-1))
            dist = pairwise_distances(wv, ent_wv).flatten()
            closest_ent_idx = dist.argsort()[0]
            closest_ent_lbl = ent_lbl[closest_ent_idx]

            res['rel_text'] = rel
            res['ent_lbl'] = closest_ent_lbl
            res['rel_lbl'] = closest_rel_lbl

            return res
        else:
            if " ".join(rel) in self.KG.rel_lbl_set:
                res['rel_lbl'] = " ".join(rel)
                res['rel_text'] = " ".join(rel)
            if ent and (ent[0] in self.KG.ent_lbl_set):
                res['ent_lbl'] = ent[0]
                
            return res
                
    
    #################### crowdsource #######################
    
    def cal_inter_rater(self, batch_idx):
        selected_batch = self.crowdsource_data[self.crowdsource_data['HITTypeId']==batch_idx]
        Pj = [round(((selected_batch['AnswerLabel']=='CORRECT').sum() / selected_batch.shape[0]),3),round(((selected_batch['AnswerLabel']=='INCORRECT').sum() / selected_batch.shape[0]),3)]
        question_idx = selected_batch['HITId'].unique().tolist()
        Pi = []
        for idx in question_idx:
            n = (selected_batch['HITId'] == idx).sum()
            pos = ((selected_batch['HITId'] == idx) & (selected_batch['AnswerLabel']=='CORRECT')).sum()
            neg = ((selected_batch['HITId'] == idx) & (selected_batch['AnswerLabel']=='INCORRECT')).sum()
            # print(f'{n,pos,neg}')
            Pi.append(round((pos*(pos-1)+neg*(neg-1))/(n*(n-1)),3))
        Po = round((sum(Pi) / len(question_idx)),3)
        Pe = sum(x**2 for x in Pj)
        return round(((Po-Pe) / (1-Pe)),3)
    
    def _crowdsource(self, extraction):
        
        ent_key = self.KG.lbl2ent[extraction['ent_lbl']]
        ent_key = self._get_label(ent_key)
        rel_key = self.KG.lbl2rel[extraction['rel_lbl']]
        rel_key = self._get_label(rel_key)
        
        if "wd:"+ ent_key in self.subject_list:
            inter_rater_score,support,reject,ans = self._crowdsourcing(ent_key, rel_key)
            response = self.generate_respond_crowdsource(extraction['ent_lbl'],
                                                        extraction['rel_text'],
                                                        ans,
                                                        inter_rater_score,
                                                        support,
                                                        reject)
            return response
        # can't solve by crowdsource
        else:
            False    
    
    

    def _crowdsourcing(self, movie_entity_key, relation_key):
        # Check whether relation exists
        if "wdt:" + relation_key in self.predicate_list:
            selected_answers = self.crowdsource_data[(self.crowdsource_data['Input1ID'] == "wd:"+ movie_entity_key) & (self.crowdsource_data['Input2ID'] == "wdt:"+relation_key)]
        else:
            selected_answers = self.crowdsource_data[self.crowdsource_data['Input1ID'] == "wd:"+ movie_entity_key]
        support = (selected_answers['AnswerLabel'] == 'CORRECT').sum()
        reject = (selected_answers['AnswerLabel'] == 'INCORRECT').sum()
        ans = selected_answers['Input3ID'].unique()[0]
        batch_idx = selected_answers['HITTypeId'].unique()[0]
        inter_rater_score = self.cal_inter_rater(batch_idx)
        if "wd:" in ans:
            ans = self.KG.all2lbl[rdflib.term.URIRef(self.KG.WD + ans.split(":")[-1])]
        # print(f'Inter-rater agreement: {inter_rater_score} Support votes: {support} Reject votes: {reject} Answer: {ans}')
        return inter_rater_score,support,reject,ans   
    
    
    def generate_respond_crowdsource(self, movie_entity, relation, ans, inter_rater_score, support, reject):
        if len(relation) > 0:
            ans = f"The {relation} of {movie_entity} is {ans}\nInter-rater agreement: {inter_rater_score} Support votes: {support} Reject votes: {reject}"
        else:
            movie_entity = movie_entity.capitalize()
            ans = f'{movie_entity} is the subclass of {ans}\nInter-rater agreement: {inter_rater_score} Support votes: {support} Reject votes: {reject}'
        return ans
    
    
    
    #################### graph_based #######################
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
            return ''
        

    def __get_word_pos(self, word) -> str:
        '''
        given a word, return pos
        '''
        return self.pos.full_tagging(word)['pos'][0] 


    def __generate_respond_graph(self, ent, rel, res):
        # if self.__get_word_pos(rel) == "NNP" or self.__get_word_pos(rel) == "NNS":
        #     ans = "The " + rel + " of " + ent + " is " + res + "."
        # else:
        #     ans = ent + " was " + rel + " " + self.__get_best_preposition(rel) + " " + res + "."
        # return ans
        return f"I think it is {res}"    
    
    def _graph_query(self, extraction):
        
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
        

        # grab result from graph
        movie_name = extraction['ent_lbl']
        target_label = extraction['rel_postfix']
        target_name = extraction['rel_lbl']
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
        for row, in self.KG.graph.query(query):
            res.append(str(row))
        # print(res)
        if len(res) != 0:
            return self.__generate_respond_graph(extraction['ent_lbl'], extraction['rel_text'], res[0])
        else:
            return False
        


    #################### embedding #######################
    def _embedding_query(self, extraction:str):
        ent_id = self.KG.ent2id[self.KG.lbl2ent[extraction['ent_lbl']]]
        rel_id = self.KG.rel2id[self.KG.lbl2rel[extraction['rel_lbl']]]
        
        head = self.KG.ent_emb[ent_id]
        pred = self.KG.rel_emb[rel_id]
        
        lhs = (head + pred).reshape((1, -1))
        
        # select closest entity
        dist = pairwise_distances(lhs, self.KG.ent_emb).flatten()
        most_likely_idx = dist.argsort()[0]
        res = self.KG.ent2lbl[self.KG.id2ent[most_likely_idx]]
        return self.__generate_respond_graph(extraction['ent_lbl'], extraction['rel_text'], res)
    
    #################### general #######################
     
    def query(self, sentence):
        
        extraction = self._model_based(sentence, similarity=False)
        print(extraction)

        # calling backup extraction strategy
        if None in extraction.values():
            print("First attempt fail, activate rule based")
            backup_extraction = self._rule_based(sentence)
            # replace None item with similarity based extraction
            for k,v in extraction.items():
                if v == None:
                    extraction[k] = backup_extraction[k]
            print(extraction)
        
        if None in extraction.values():
            print("Second attempt fail, activate similarity based")
            backup_extraction = self._model_based(sentence, similarity=True)
            # replace None item with similarity based extraction
            for k,v in extraction.items():
                if v == None:
                    extraction[k] = backup_extraction[k]    
            print(extraction)
            
        extraction['rel_postfix'] = self._get_label(self.KG.lbl2rel[extraction['rel_lbl']])
        
        # make sure there are no None values
        if None in  extraction.values():
            return "sorry, I'm not able to understand your question now. Could you please reorgnize you sentence? Try to contain relation and entity name this time ;)"
        
        # try to solve by crowdsource
        response = self._crowdsource(extraction)
        if response:
            print(extraction)
            return response
        
        # try to solve by graph query
        response = self._graph_query(extraction)
        if response:
            print(extraction)
            return response
        
        # try to solve by embedding
        response = self._embedding_query(extraction)
        print(extraction)
        return "I can't get any knowledge of it in my knowledge graph, but according to the embedding, " + response
        

    
if __name__=='__main__':
    import sparknlp
    spark = sparknlp.start(gpu=True)
    testing_case = 'What is the box office of The Princess and the Frog?'
    
    ner = LM_ner()
    KG = KG_handler()
    pos = LM_pos()
    
    quer = fact_querier(KG, pos, ner)
    print(quer.query(testing_case))
    
from speakeasypy import Speakeasy
import time
import os
from collections import Counter

# import spacy
# from spacy.lang.en import English
import nltk
from nltk.corpus import brown
import sparknlp
from LM import *
from KG_handler import *
from Recommendor import *
from Multimedia import *
from Q_classifier import *
from Quer import *


# import re
# import editdistance
# import pickle

# import cv2
# import logging

USER_NAME = 'broil-grandioso-pie_bot'
PWD = 'NrBl_CRF6101Dg'
HOST = 'https://speakeasy.ifi.uzh.ch'
    
    
class response_generator():
    def __init__(self, KG):
        self.reco_dict = {
            'person_templates' : {
                "default": "movies involing {name} during producing"
            },

            'genre_templates' : {
                "action": "an adrenaline-fueled {genre} with explosive action",
                "comedy": "a hilarious {genre} filled with laughter",
                "sci-fi": "explore {genre} worlds and mind-bending concepts",
                "horror": "a spine-chilling {genre} that will haunt your dreams",
                "romance": "a heartfelt {genre} with passion and emotion",
                "fantasy": "escape to enchanting {genre} in a magical adventure",
                "mystery": "unravel {genre} secrets and solve mysteries",
                "thriller": "an intense {genre} experience that will keep you on edge"
            },

            'emotion_templates' : {
                "romantic": "get ready for a romantic journey",
                "love": "love takes center stage in this heartfelt story",
                "heartwarming": "experience a heartwarming tale that leaves you smiling",
                "uplifting": "an uplifting story that inspires and fills the heart with joy",
                "tear-jerker": "prepare for an emotional rollercoaster",
                "bittersweet": "a bittersweet narrative balancing love and loss",
                "feel-good": "a feel-good movie that lifts your spirits",
                "passionate": "feel the passion in every scene of this emotionally charged film"
            },

            'default_templates' : {
                "default": "movies characterized by {default}"
            }
        }    
        
        self.KG = KG
        
    
    def generate(self, args):
        return self.__reco_generation(args)
            
    
    def __reco_generation(self, args):
        features = args[0]
        movie = args[1]
        
        description = []
        response = []
        for feature in features:
            # If the feature is Name
            # if isinstance(feature, list):
            #     template_key, feature_name = feature
            #     if template_key in self.reco_dict['person_templates']:
            #         description.append(self.reco_dict['person_templates'][template_key].format(feature=feature_name))
            #     else:
            #         description.append(self.reco_dict['default_templates']["default_person"].format(feature=feature_name))
            
            if self.KG._is_human(feature):
                description.append(self.reco_dict['person_templates']['default'].format(name=feature))
            elif feature in self.reco_dict['genre_templates']:
                description.append(self.reco_dict['genre_templates'][feature].format(genre=feature))
            elif feature in self.reco_dict['emotion_templates']:
                description.append(self.reco_dict['emotion_templates'][feature])
            else:
                description.append(self.reco_dict['default_templates']["default"].format(default=feature))
        # print(description)
        if description:
            or_joined = " or ".join(description[:len(description)//2])
            comma_joined = ", ".join(description[len(description)//2:])
            response.append(f"Adequate recommendations will be {or_joined}. Alternatively, you might enjoy {comma_joined}.")
            if movie:
                response.append("Base on my knowledge, I hereby recommend following movies for you: %s"%str(movie))
            return ' '.join(response)
        else:
            response.append("No specific information available for the provided movie features.")
            if movie:
                response.append("But according to my content-based analysis, I can still recommend you some movies: %s"%str(movie))
                return ' '.join(response)
            else:
                return "Oops...Unfortunately I failed to recommend anything for you. Please make sure you typed all movie names correctly. Notice the letter case ;)"
            
        
        
        
    
class Agent:
    def __init__(self, user_name=USER_NAME, pwd=PWD, host=HOST, log_path = 'logs'):
        
        # activate GPU

        
        self.user_name = user_name
        self.pwd = pwd
        self.host = host
        self.activated_room = []
        
        try:
            self.spark = sparknlp.start(gpu=True)
        except:
            self.spark = sparknlp.start()
            
        self.LM_pos = LM_pos()
        self.LM_ner = LM_ner()
        self.KG_handler = KG_handler()
        
        self.Recommendor = recommender(self.KG_handler, self.LM_pos, self.LM_ner)
        self.querier = fact_querier(self.KG_handler, self.LM_pos, self.LM_ner)
        self.multimeida = multimedia_handler(self.KG_handler, self.LM_ner)
        self.resp_generator = response_generator(self.KG_handler)
        self.classifier = Q_classifier()
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
        
        
        nltk.download("universal_tagset")
        nltk.download("brown")

        verb_prep_pairs = [(v[0].lower(), p[0].lower()) 
                         for (v, p) in nltk.bigrams(brown.tagged_words(tagset="universal")) 
                         if v[1] == "VERB" and p[1] == "ADP"]
        self.verb_prep_counts = Counter(verb_prep_pairs)
        # self.little_tiny_tagging = spacy.load("en_core_web_sm")
        
        
    def __query(self, query:str) -> str:
        # Judge whether the query is in Natural Language or SPARQL
        if "SELECT" in query:
            res = self.__query__sparsql(query)
            return res
        else:
            print("Query is in NL style!")
            self.doc = self.spacy_model(query)
            extraction, res = self.KG_handler.get_query_res(query)
            if res == False:
                return "Sorry :( I fail to find the answer for. I will learn harder and be more intelligent! Meet u in the future ~"
            print("Final extraction")
            print(extraction)
            print(res)
            return self.__return_ans(movie_name=extraction['ent_lbl'],
                                    relation=extraction['rel'],
                                    res=res)
        
    
        
        
    # def __get_best_preposition(self, verb):
    #     '''
    #     given verb, return the proper prep
    #     '''
    #     verb = verb.lower()
    #     prepositions = [(prep, count) for (v, prep), count in self.verb_prep_counts.items() if v == verb]
    #     if prepositions:
    #         best_preposition = max(prepositions, key=lambda x: x[1])
    #         return best_preposition[0]
    #     else:
    #         return None



    # def __get_word_pos(self, word) -> str:
    #     '''
    #     given a word, return pos
    #     '''
    #     doc = self.LM_pos.full_tagging(word) 
    #     print('doc:', doc)
    #     pos_tag = doc[0].pos_  
    #     return pos_tag



    # def __return_ans(self, movie_name, relation, res) -> str:
    #     '''
    #     construct answer with movie name, relation and query result
    #     '''

    #     # Temporarily fix unicode issue
    #     if '–' in movie_name:
    #         movie_name = movie_name.replace('–','-')

    #     if self.__get_word_pos(relation) == "NOUN":
    #         ans = "The " + relation + " of " + movie_name + " is " + res + "."
    #     else:
    #         # ans = movie_name + " was " + relation + " " + self.__get_best_preposition(relation) + " " + res + "."
    #         preposition = self.__get_best_preposition(relation) # preposition may be empty which is decided by the pre-trained model
    #         if preposition != None:
    #             ans = movie_name + " was " + relation + " " + preposition + " " + res + "."
    #         else:
    #             ans = movie_name + " was " + relation + " " + res + "."
    #     return ans

    
    def __query__sparsql(self, query:str) -> list:
        # preprocessing
        query = query.strip(" ")
        query = query.strip("'")
        
        res = []
        for row in self.graph.query(query):
            res.append([str(i) for i in row])
        
        return res
    
    
    def __question_classifier(self, sentence:str) -> int:
        
        question_type = self.classifier.question_classify(sentence)
        
        return question_type
    
    def __real_time_logging(self, msg:str):
        time_format = time.strftime("[%Y-%m-%d %H:%M:%S]\n", time.localtime())
        with open(self.log_path, mode='a', encoding='utf-8') as f:
            f.write(time_format)
            f.write(msg)
        
    def start(self):
        print('start')
        while True:
            rooms = self.chat_agent.get_rooms(active=True)
            for room in rooms:          # check whether there is new room 
                room_id = room.room_id

                if room.room_id not in self.activated_room:         
                    self.activated_room.append(room_id)
                    self.__real_time_logging("New chat room started: %s"%room_id)
                
                # Retrieve messages from this chat room.
                for message in room.get_messages(only_partner=True, only_new=True):
                    
                    self.__real_time_logging("From room %s\n Received message %s"%(room_id, message.message))
                    # print("msg: ", message.message)
                    # try:
                    #     ans = self.__query(str(message.message))     
                    # except:
                    #     ans = 'Null, query fail or error happened'
                    sentence = str(message.message)
                    question_type = self.__question_classifier(sentence)
                    print(question_type)
                    try:
                    
                        if question_type == 0:
                            ans = self.querier.query(sentence=sentence)
                        elif question_type == 1:
                            ans = self.multimeida.show_img(sentence=sentence)
                        elif question_type == 2:
                        
                            ans1 = self.Recommendor.recommend(sentence=sentence, mode='feature')
                            ans2 = self.Recommendor.recommend(sentence=sentence, mode='movie')
                            print(ans1)
                            print(ans2)
                            ans = self.resp_generator.generate([ans1, ans2])
                    
                    except:
                        ans = "Oops, there is something wrong with me. Can you give another question?"
                        
                    # print('ans: ', ans) 
                
                    ans = ans.encode('utf-8')
                    room.post_messages(ans.decode('latin-1'))
                    # logging.info("To room %s\n Reply message: %s"%(room_id, ans))
                    # self.__real_time_logging("To room %s\n Reply message \{%s\}"%(room_id, ans))

                    room.mark_as_processed(message)
                    
                    
                # Retrieve reactions from this chat room.
                for reaction in room.get_reactions(only_new=True):
                    # Implement your agent here #
                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)
                    
if __name__ == '__main__':
    bot = Agent()
    bot.start()                    
from speakeasypy import Speakeasy
import rdflib
import time
import os
import keyboard
# import cv2
# import logging

USER_NAME = 'broil-grandioso-pie_bot'
PWD = 'NrBl_CRF6101Dg'
HOST = 'https://speakeasy.ifi.uzh.ch'

class Agent:
    def __init__(self, user_name=USER_NAME, pwd=PWD, host=HOST, graph_path='graph/14_graph.nt', log_path = 'logs'):
        self.user_name = user_name
        self.pwd = pwd
        self.host = host
        self.graph_path = graph_path
        self.activated_room = []
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
        
                
        # initialize RDF
        self.graph = rdflib.Graph()
        self.graph.parse(self.graph_path, format='turtle')
        print("Initializing RDF done!")
        
        
        
        
        
    def __query(self, query:str) -> list:
        # preprocessing
        query = query.strip(" ")
        query = query.strip("'")
        
        res = []
        for row in self.graph.query(query):
            res.append([str(i) for i in row])
        
        return res
    
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
                        ans = 'Null'
                        
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
                    
            # exiting detect
            if keyboard.is_pressed('q'):
                if input("Detect 'q', do you wannt to exit? [y/n]:")=='y':
                    print('killing program...')
                    break
                    
if __name__ == '__main__':
    bot = Agent()
    bot.start()                    
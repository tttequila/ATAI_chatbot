from KG_handler import *
from LM import *
import json

class multimedia_handler():
    def __init__(self, KG_handler:KG_handler, LM_ner:LM_ner, json_path='utils/images.json'):
        self.KG = KG_handler
        self.ner = LM_ner
        
        with open(json_path, 'r') as f:
            self.image_net = json.load(f)   
        self.imgs = [i['img'] for i in self.image_net]
        self.ids = [set(i['movie']+i['cast']) for i in self.image_net]
        
        self.PANELTY = 0.3

        
    def ent_to_id(self, entities):
        ent_dic = {}
        for ent_name in entities:
            query = '''PREFIX ddis: <http://ddis.ch/atai/>

            PREFIX wd: <http://www.wikidata.org/entity/>

            PREFIX wdt: <http://www.wikidata.org/prop/direct/>

            PREFIX schema: <http://schema.org/>

            SELECT ?obj WHERE {
                SERVICE <https://query.wikidata.org/sparql>{
                ?sub rdfs:label "%s"@en.

                ?sub wdt:P345 ?obj.
            }
            } LIMIT 1'''%(ent_name)
            query  = query.strip()

            ent_dic[ent_name] = []
            for row, in self.KG.graph.query(query):
            # ent_dic[ent_name] = [str(ent) for ent, in graph.query(query)]
                ent_dic[ent_name].append(str(row))

        
        tmp = []
        for ent_list in ent_dic.values():
            tmp += ent_list
        return tmp

    def show_img(self, sentence):
        entities = self.ner.extraction(sentence)
        print('=====================', entities)
        id_lst = self.ent_to_id(entities)
        print('=====================', id_lst)
        score_lst = [len(set(id_lst) & single_img) - self.PANELTY *len(single_img) for single_img in self.ids]
        idx = np.argmax(score_lst)
        print('=====================', len(score_lst), idx)
        
        return 'image:%s'%str(self.imgs[idx].split('.')[0])
    

if __name__ == "__main__":
    import sparknlp
    spark = sparknlp.start(gpu=True)
    testing_case = 'i wanna see the poster of Julia Roberts in Pretty Woman'

    ner = LM_ner()
    graph = KG_handler() 
    
    multimedia = multimedia_handler(graph, ner)
    print(multimedia.show_img(testing_case))

        
        

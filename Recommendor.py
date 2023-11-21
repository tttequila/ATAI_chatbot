from KG_handler import *
from LM import *
from sklearn.cluster import KMeans
# import editdistance
from sklearn.metrics import pairwise_distances

class recommender():
    def __init__(self, KG_handler:KG_handler, LM_pos:LM_pos, LM_ner:LM_ner):
        
        '''
        Params:
            graph: KG_handler object
            LM_pos: pos language model object
            LM_ner: ner language model object            
        '''
        self.KG = KG_handler
        self.ner = LM_ner
        self.pos = LM_pos
        
        self.common_feature = []
        self.movies = pickle.load(open('end2end/movies.pkl', 'rb'))
        self.similarity = pickle.load(open('end2end/similarity.pkl', 'rb'))
        with open("utils/movie_list.txt", 'r', encoding="UTF-8") as f:
            self.movie_name_list = f.readlines()
        self.movie_name_list = [movie.replace('\n', '') for movie in self.movie_name_list]
        # with open("utils/predifined_rel_name.txt",  'w', encoding="UTF-8") as f:
        #     self.rel_name = f.rea
        self.rel_name_list = [
            'film editor', 'genre', 'nominated for', 'published in', 'award received', 'part of the series',
            'production designer', 'production company', 'art director', 'musical conductor',
            'original film format', 'platform', 'derivative work', 'sound designer', 'screenwriter',
            'character designer', 'inspired by', 'main subject', 'distributed by', 'followed by', 'genre'
            ]
        self.rel_list = [self.KG.lbl2rel[rel.replace('\n', '')] for rel in self.rel_name_list]

    
    def recommend(self, sentence:str, mode:"str in ['movie', 'feature']") -> str:
        '''
        Param:
            sentence: get natural language as input
        Return:
            reco_sentence (str | list(str)) havn't decide which return yet really depands of our answer generator
        '''
        assert mode in ['movie', 'feature']
        if mode == 'movie':
            return self._movie_reco(sentence)
        if mode == 'feature':
            return self._common_feature(sentence)
        
    def _common_feature(self, sentence:str, K=10) -> list:
        '''
        Params:
            sentence: 
            K: top-K common features
        Return:
            movies: a list of movie Named entities in the sentence
            closest_rel_lbl: a list of common feature given movie list (currently only entities, but we can attach relation later)
        '''
        movies = self.ner.extraction(sentence)
        movies = [movie for movie in movies if movie in self.movie_name_list]
        
        query_compo = []
        features_emb = []
        for rel in self.rel_list:
            query_compo.append("{?movie <%s> ?obj . }" % rel)
            # union_parts.append("{?obj <%s> ?movie . }" % rel)
            
        union_query = " UNION ".join(query_compo)
        for movie_name in movies:
            query = '''SELECT ?obj
                        WHERE {
                            
                            SERVICE <https://query.wikidata.org/sparql>{
                            
                                ?movie rdfs:label "%s"@en. 
                                                
                                %s                              

                            }
                        }'''%(movie_name, union_query)
            query = query.strip()
        
            res = []
            for row in self.KG.graph.query(query):
                res.append([str(i) for i in row]) 
            emb_list = [self.KG.ent_emb[self.KG.ent2id[rdflib.term.URIRef(ent[0])]] 
                            for ent in res if rdflib.term.URIRef(ent[0]) in self.KG.ent2id.keys()]
            features_emb += emb_list
        cluster = KMeans(n_clusters=int(len(features_emb)/2), n_init='auto')
        cluster.fit(np.array(features_emb))
        labels_cnt = np.bincount(cluster.labels_)

        top_label_idx = np.argsort(labels_cnt)[::-1][:K]
        centroids = cluster.cluster_centers_[top_label_idx]
        dist = pairwise_distances(centroids, self.KG.ent_emb)
        
        
        # select features based on the most K-th significant centroids 
        closest_rel_idx = dist.argsort()[:,0]
        # closest_rel_lbl = [id2ent[i] for i in closest_rel_idx]
        closest_rel_uri = [self.KG.id2ent[i] for i in closest_rel_idx]
        closest_rel_lbl = [self.KG.ent2lbl[i] for i in closest_rel_uri]
        
        return closest_rel_lbl

    def _recom(self, movie):
        movie_index = self.movies[self.movies['title'] == movie].index[0]
        distances = self.similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        for i in movies_list:
            recommended_movies.append(self.movies.iloc[i[0]]['title'])
        return recommended_movies
    
    def _movie_reco(self, sentence) -> list:
        '''
        Params:
            sentence
        Return:
            movies: list of movie named entities in the sentence
            recom_movies: recommended movie according to content-based recommendation 
            
        '''
        movies = self.ner.extraction(sentence)
        movies = [movie for movie in movies if movie in self.movie_name_list]

        recom_movies = []
        for movie in movies:
            try:
                recom_movies.append(self._recom(movie))
            except:
                "Can not find related movie in the dataset"

        return movies, recom_movies
    
if __name__ == '__main__':
    import sparknlp
    # import pyspark
    # pyspark.SparkContext().setLogLevel(logLevel="INFO")
    
    sentence = 'Given that I like The Lion King, Pocahontas, and The Beauty and the Beast, can you recommend some movies?'
    
    spark = sparknlp.start(gpu=True)
    pos = LM_pos()
    ner = LM_ner()
    graph = KG_handler() 
    reco_sys = recommender(graph, pos, ner)
    
    pos = LM_pos().full_tagging(sentence)
    
    # features = reco_sys._common_feature(sentence)
    print(pos)
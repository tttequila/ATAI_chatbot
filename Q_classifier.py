import pickle
from keras.utils import pad_sequences
from tensorflow.keras.models import load_model

class Q_classifier():
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 20
        
        self.FAC = 0
        self.MUL = 1
        self.REC = 2
        
        self.label_list = {0:"fac",1:"mul",2:"rec"}
        
        with open('utils/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        self.model = load_model("utils/intent classification.h5")
        
    def question_classify(self, sentence):
        sequences_new = self.tokenizer.texts_to_sequences([sentence])
        data = pad_sequences(sequences_new, maxlen=self.MAX_SEQUENCE_LENGTH)
        yprob = self.model.predict(data)
        yclasses = yprob.argmax(axis=-1)
        return yclasses[0]
    
    
if __name__ == '__main__':
    classifier = Q_classifier()
    sentence = 'Who directed The Bridge on the River Kwai?'
    print(classifier.question_classify(sentence))

from transformers import pipeline

from senta import Senta

class SentenceSentiment_Analyzer():
    def __init__(self, name, device):
        self.name = name
        if name == "BaiduAPI":
            self.analyzer = Senta()
            self.analyzer.init_model(model_class="ernie_1.0_skep_large_ch", task="sentiment_classify", use_cuda=(device != "cpu"))
        elif name == "HuggingFace_techthiyanes":
            self.analyzer = pipeline('sentiment-analysis', model = "techthiyanes/chinese_sentiment", device = int(device.split(":")[1]))

    def predict(self, sentence):
        if self.name == "BaiduAPI":
            vs = self.analyzer.predict(sentence)[0]
            # print(vs)
            if vs[1] == 'positive':
                return 1
            else:
                return -1
            # return self.analyzer.predict(sentence)
        elif self.name == 'HuggingFace_techthiyanes':
            vs = self.analyzer(sentence)[0]['label']
            if vs == 'star 5':
                return 1
            elif vs == 'star 1':
                return -1
            else:
                return 0
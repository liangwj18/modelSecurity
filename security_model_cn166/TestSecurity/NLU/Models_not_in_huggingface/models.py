import importlib
import ERNIE.code.knowledge_bert as knowledge_bert
# from .ERNIE.code import knowledge_bert

def LoadModel():
    # model = importlib.import_module(".ERNIE.code.knowledge_bert", __package__)
    model = knowledge_bert.BertForMaskedLM.from_pretrained('ernie_base')
    tokenizer = knowledge_bert.BertTokenizer.from_pretrained('ernie_base')
if __name__ == "__main__":
    LoadModel()
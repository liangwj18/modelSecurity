import paddle
from paddlenlp.embeddings import TokenEmbedding

token_embedding = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-ngram.1-2.dim300")
vocab = token_embedding.vocab #获得词汇表
print(vocab['赞成'])
print(vocab['同意'])
print(vocab['是的'])

w2v = token_embedding.weight
#打印词向量的形状大小
print(w2v.shape)
#根据索引打印某个词对应的词向量
print(w2v[2965])

print(token_embedding.cosine_sim("同意","是的"))

#使用paddle函数计算余弦相似度
print(paddle.nn.functional.cosine_similarity(w2v[2965],w2v[3886],axis=0))
print(token_embedding.cosine_sim("是的","不是"))
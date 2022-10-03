
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans


one = open(r'projs.txt',encoding = "utf-8")
onee = list(one)
one.close()

if __name__ == "__main__":

    # Question a
    corpus= onee
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    word=vectorizer.get_feature_names()
    weight=tfidf.toarray()
    for i in range(len(weight)):
        print(u"list",i+1,"-----")
        for j in range(len(word)):
            if(weight[i][j] >= 0.01):
                print(word[j],weight[i][j])

    # print(weight.shape) (22, 108)

    # # Question b

    weight = torch.Tensor(weight)
    my = weight[-2]
    sim_list = []
    for i in range(weight.shape[0]):
        cos = F.cosine_similarity(weight[i], my, dim =0)
        if cos > 0.9:
            sim_list.append(i)
    print(sim_list)
    # # Question c

    kmean = KMeans(n_clusters=6).fit(weight)

    print(kmean.labels_)




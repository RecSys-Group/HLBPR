import numpy as np
from sklearn.decomposition import ProjectedGradientNMF
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as spr
from Metrics import BasicMetric
import math
import random
class NMF_MF:
    def __init__(self):
        print "TopN begin"
    def load_data(self, users, items, trainfile, testfile):
        print 'loading data...'
        self.file_users = users
        self.file_items = items
        self.trainfile = trainfile
        self.testfile = testfile
        self.users = pd.read_csv(self.file_users)
        self.items = pd.read_csv(self.file_items)
        self.train = pd.read_csv(self.trainfile, dtype='str')
        self.test = pd.read_csv(self.testfile, dtype='str')
        self.n_features = 50
        temp = math.sqrt(self.n_features)
        self.qi = [[(0.1 * random.random() / temp) for j in range(self.n_features)] for i in range(len(self.items))]
        self.pu = [[(0.1 * random.random() / temp) for j in range(self.n_features)] for i in range(len(self.users))]
    def get_trunk(self, goods):
        length = len(goods)
        for i in range(length):
            if goods[i].find('-') > 0:
                break
        return goods[1:i - 1]
    def new_get_trunk(self, row):
        for i in range(len(row)):
            if math.isnan(float(row[i])):
                break
        return row[:i]
    def from_product_to_pid(self, product):
        products = np.array(self.items.iloc[:, 1])
        pid = np.argwhere(products == int(product))
        if len(pid) != 0:
            return pid[0][0]
        else:
            return -1
    def from_user_to_uid(self, user):
        print user
        users = np.array(self.users.iloc[:, 1])
        uid = np.argwhere(users == int(user))
        return uid[0][0]
    def gen_recommendation(self, uid, N):
        predict_scores = []
        for i in range(len(self.items)):
            s = self.InerProduct(self.pu[uid], self.qi[i])
            predict_scores.append(s)
        topN = np.argsort(np.array(predict_scores))[-1:-N - 1:-1]
        return topN
    def get_user_purchased(self, uid):
        index = np.argwhere(np.array(self.train.iloc[:, 1]) == str(float(uid)))
        index_p = [i[0] for i in index]
        products = list(np.array(self.train.iloc[:, 2])[index_p])
        result = [int(float(i)) for i in products]
        return result
    def InerProduct(self, v1, v2):
        result = 0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result
    def ValidateF1(self):
        print 'evaluating...'
        recommendation_list = []
        purchased_list = []
        metric = BasicMetric.BasicMetrics()
        for row in self.test.values:
            prodeucts = self.new_get_trunk(row)[1:]
            pids = [self.from_product_to_pid(int(p)) for p in prodeucts]
            purchased_list.append(pids)
            uid = self.new_get_trunk(row)[0]
            recom = self.gen_recommendation(int(uid), 5)
            recommendation_list.append(recom)
            print pids, recom
        p, r, f, h = metric.F1_score_Hit_ratio(recommendation_list, purchased_list)
        ndgg = metric.NDGG_k(recommendation_list, purchased_list)
        t = pd.DataFrame(np.array(['F_1s are :' + str(f)+'Hit_ratios are :' + str(h)]))
        t.to_csv('50f1')
        t = pd.DataFrame(np.array(['NDGG are :' + str(ndgg)]))
        t.to_csv('50ndgg')
        return f
    def train_model(self):
        print 'begin'
        RATE_MATRIX = np.zeros((9238, 7973))
        for line in self.train.values:
            print line
            uid = int(float(line[1]))
            iid = int(float(line[2]))
            RATE_MATRIX[uid][iid] = int(float(line[3]))
        V = spr.csr_matrix(RATE_MATRIX)
        model = ProjectedGradientNMF(n_components=self.n_features, max_iter=1000, nls_max_iter=10000)
        self.pu = model.fit_transform(V)
        self.qi = model.fit(V).components_.transpose()
        print model.reconstruction_err_
        self.ValidateF1()
        t = pd.DataFrame(np.array(self.pu))
        t.to_csv('50pu')
        t = pd.DataFrame(np.array(self.qi))
        t.to_csv('50qi')
        print("model generation over")
def start(users, items, trainfile, testfile):
    nmf = NMF_MF()
    nmf.load_data(users, items, trainfile, testfile)
    nmf.train_model()
if __name__ == '__main__':
    start()

    









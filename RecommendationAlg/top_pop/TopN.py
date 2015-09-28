import numpy as np
import pandas as pd
from Metrics import BasicMetric
import math
import os

class TopN:
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
        self.test = pd.read_csv(self.testfile, dtype='str')
        self.train = pd.read_csv(self.trainfile, dtype='str')
    def gen_items_popular(self,popfile):
        print 'gen_popular!'
        if os.path.exists(popfile):
            print 'items_popularity has been generated!'
        else:
            itempopular = np.zeros(len(self.items))
            for row in self.train.values:
                print row
                iid = int(float(row[2]))
                times = int(float(row[3]))
                itempopular[iid] += times
            t = pd.DataFrame(itempopular)
            t.to_csv(popfile)
    def make_predictions(self, N, popfile):
        self.popfile = popfile
        self.itempopular = pd.read_csv(self.popfile)
        self.topN = np.argsort(np.array(self.itempopular.iloc[:, 1]))[-1:-N-1:-1]
        print self.topN
        return self.topN
    def get_trunk(self, goods):
        length = len(goods)
        for i in range(length):
            if goods[i].find('-') > 0:
                break
        return goods[1:i-1]
    def new_get_trunk(self, row):
        for i in range(len(row)):
            if math.isnan(float(row[i])):
                break
        return row[:i]
    def from_product_to_pid(self, product):
        products = np.array(self.items.iloc[:, 1])
        pid = np.argwhere(products == product)
        if len(pid) != 0:
            return pid[0][0]
        else:
            return -1
    def evaluate(self):
        print 'evaluating...'
        recommendation_list = []
        purchased_list = []
        metric = BasicMetric.BasicMetrics()
        for row in self.test.values:
            recommendation_list.append(list(self.topN))
            prodeucts = self.new_get_trunk(row)
            pids = [self.from_product_to_pid(int(p)) for p in prodeucts]
            purchased_list.append(pids)
        metric.F1_score_Hit_ratio(recommendation_list, purchased_list)
        metric.NDGG_k(recommendation_list, purchased_list)
def start(users, items, trainfile, testfile, popfile):
    tn = TopN()
    tn.load_data(users, items, trainfile, testfile)
    tn.gen_items_popular(popfile)
    tn.make_predictions(5, popfile)
    tn.evaluate()

if __name__ == '__main__':
    start()



import numpy as np
from math import exp
import random
from scipy.sparse import coo_matrix
import scipy.sparse as spr
import pandas as pd
import math
from Metrics import BasicMetric

class BPRArgs(object):
    def __init__(self,learning_rate=0.005,
                 bias_regularization=0.0,
                 user_regularization=0.002,
                 positive_item_regularization=0.0003,
                 negative_item_regularization=0.0003,
                 update_negative_item_factors=True):
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.update_negative_item_factors = update_negative_item_factors
class BPR(object):
    def __init__(self,D,args,users, items, trainfile, testfile, pu, qi):
        self.D = D
        self.learning_rate = args.learning_rate
        self.bias_regularization = args.bias_regularization
        self.user_regularization = args.user_regularization
        self.positive_item_regularization = args.positive_item_regularization
        self.negative_item_regularization = args.negative_item_regularization
        self.update_negative_item_factors = args.update_negative_item_factors
        self.file_users = users
        self.file_items = items
        self.testfile = testfile
        self.trainfile = trainfile
        self.initpu = pu
        self.initqi = qi
        self.cxusers = pd.read_csv(self.file_users)
        self.cxitems = pd.read_csv(self.file_items)
        self.cxtrain = pd.read_csv(self.trainfile, dtype='str')
        self.cxtest = pd.read_csv(self.testfile, dtype='str')
    def train(self,data,sampler,num_iters):
        self.init(data)
        old_loss = self.loss()
        print 'initial loss = {0}'.format(self.loss())
        for it in xrange(num_iters):
            print 'starting iteration {0}'.format(it)
            for u, i, j in sampler.generate_samples(self.data):
                self.update_factors(u, i, j)
            if abs(self.loss() - old_loss) < 1:
                print 'iteration {0}: loss = {1}'.format(it, self.loss())
                print 'converge!!'
                break
            else:
                old_loss = self.loss()
                self.learning_rate *= 0.9
                print 'iteration {0}: loss = {1}'.format(it, self.loss())
        self.ValidateF1()
        t = pd.DataFrame(self.user_factors)
        t.to_csv('bpr50pu')
        t = pd.DataFrame(self.item_factors)
        t.to_csv('bpr50qi')
        t = pd.DataFrame(self.item_bias)
        t.to_csv('bpr50b')
    def init_user_item(self):
        user_file = self.initpu
        item_file = self.initqi
        user_data = pd.read_csv(user_file)
        item_data = pd.read_csv(item_file)
        return np.array(user_data.values)[:, 1:], np.array(item_data.values)[:, 1:]
    def init(self, data):
        self.data = data
        self.num_users, self.num_items = self.data.shape
        self.item_bias = np.zeros(self.num_items)
        self.user_factors, self.item_factors = self.init_user_item()
        self.create_loss_samples()
    def create_loss_samples(self):
        num_loss_samples = int(100*self.num_users**0.5)
        print 'sampling {0} <user,item i,item j> triples...'.format(num_loss_samples)
        sampler = UniformUserUniformItem(True)
        self.loss_samples = [t for t in sampler.generate_samples(self.data, num_loss_samples)]
    def update_factors(self, u, i, j, update_u=True, update_i=True):
        update_j = self.update_negative_item_factors
        x = self.item_bias[i] - self.item_bias[j] \
            + np.dot(self.user_factors[u,:],self.item_factors[i,:]-self.item_factors[j,:])
        z = 1.0/(1.0+exp(x))
        if update_i:
            d = z - self.bias_regularization * self.item_bias[i]
            self.item_bias[i] += self.learning_rate * d
        if update_j:
            d = -z - self.bias_regularization * self.item_bias[j]
            self.item_bias[j] += self.learning_rate * d

        if update_u:
            d = (self.item_factors[i,:]-self.item_factors[j,:])*z - self.user_regularization*self.user_factors[u,:]
            self.user_factors[u,:] += self.learning_rate*d
        if update_i:
            d = self.user_factors[u,:]*z - self.positive_item_regularization*self.item_factors[i,:]
            self.item_factors[i,:] += self.learning_rate*d
        if update_j:
            d = -self.user_factors[u,:]*z - self.negative_item_regularization*self.item_factors[j,:]
            self.item_factors[j,:] += self.learning_rate*d
    def loss(self):
        ranking_loss = 0
        for u,i,j in self.loss_samples:
            x = self.predict(u,i) - self.predict(u,j)
            ranking_loss += math.log(1.0+exp(-x))
        complexity = 0
        for u,i,j in self.loss_samples:
            complexity += self.user_regularization * np.dot(self.user_factors[u],self.user_factors[u])
            complexity += self.positive_item_regularization * np.dot(self.item_factors[i],self.item_factors[i])
            complexity += self.negative_item_regularization * np.dot(self.item_factors[j],self.item_factors[j])
            complexity += self.bias_regularization * self.item_bias[i]**2
            complexity += self.bias_regularization * self.item_bias[j]**2
        return ranking_loss + 0.5*complexity
    def new_get_trunk(self, row):
        for i in range(len(row)):
            if math.isnan(float(row[i])):
                break
        return row[:i]
    def from_product_to_pid(self, product):
        products = np.array(self.cxitems.iloc[:, 1])
        pid = np.argwhere(products == int(product))
        if len(pid) != 0:
            return pid[0][0]
        else:
            return -1
    def from_user_to_uid(self, user):
        users = np.array(self.cxusers.iloc[:, 1])
        uid = np.argwhere(users == int(user))
        return uid[0][0]
    def gen_recommendation(self, uid, N):
        predict_scores = []
        for i in range(len(self.cxitems)):
            s = self.predict(uid, i)
            predict_scores.append(s)
        topN = np.argsort(np.array(predict_scores))[-1:-N - 1:-1]
        return topN
    def get_user_purchased(self, uid):
        index = np.argwhere(np.array(self.cxtrain.iloc[:, 1]) == str(float(uid)))
        index_p = [i[0] for i in index]
        products = list(np.array(self.cxtrain.iloc[:, 2])[index_p])
        result = [int(float(i)) for i in products]
        return result
    def ValidateF1(self):
        print 'evaluating...'
        recommendation_list = []
        purchased_list = []
        metric = BasicMetric.BasicMetrics()
        for row in self.cxtest.values:
            prodeucts = self.new_get_trunk(row)[1:]
            pids = [self.from_product_to_pid(int(p)) for p in prodeucts]
            purchased_list.append(pids)
            uid = self.new_get_trunk(row)[0]
            recom = self.gen_recommendation(int(uid), 5)
            recommendation_list.append(recom)
            print pids, recom
        p, r, f, h = metric.F1_score_Hit_ratio(recommendation_list, purchased_list)
        ndgg = metric.NDGG_k(recommendation_list, purchased_list)
        t = pd.DataFrame([str(f), str(h), str(ndgg)])
        t.to_csv('bpr50_result')
        return f
    def predict(self,u,i):
        return self.item_bias[i] + np.dot(self.user_factors[u],self.item_factors[i])
class Sampler(object):

    def __init__(self,sample_negative_items_empirically):
        self.sample_negative_items_empirically = sample_negative_items_empirically

    def init(self,data,max_samples=None):
        self.data = data
        self.num_users,self.num_items = data.shape
        self.max_samples = max_samples

    def sample_user(self):
        u = self.uniform_user()
        num_items = self.data[u].getnnz()
        assert(num_items > 0 and num_items != self.num_items)
        return u

    def sample_indicate_item(self, items_list):
        j = self.random_item()
        while j not in items_list:
            j = self.random_item()
        return j

    def sample_negative_item(self,user_items):
        j = self.random_item()
        while j in user_items:
            j = self.random_item()
        return j

    def uniform_user(self):
        return random.randint(0,self.num_users-1)

    def random_item(self):
        if self.sample_negative_items_empirically:
            u = self.uniform_user()
            i = random.choice(self.data[u].indices)
        else:
            i = random.randint(0,self.num_items-1)
        return i
    def num_samples(self,n):
        if self.max_samples is None:
            return n
        return min(n,self.max_samples)
class UniformUserUniformItem(Sampler):
    def get_user_purchased(self, data, uid):
        index = np.argwhere(np.array(data.iloc[:, 1]) == int(uid))
        index_p = [i[0] for i in index]
        products = list(np.array(data.iloc[:, 2])[index_p])
        result = [int(float(i)) for i in products]
        return result
    def generate_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        for _ in xrange(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            i = random.choice(self.data[u].indices)
            j = self.sample_negative_item(self.data[u].indices)
            yield u, i, j
class UniformPairWithoutReplacement(Sampler):
    def get_user_purchased(self, data, uid):
        index = np.argwhere(np.array(data.iloc[:, 1]) == int(uid))
        index_p = [i[0] for i in index]
        products = list(np.array(data.iloc[:, 2])[index_p])
        result = [int(float(i)) for i in products]
        return result
    def generate_samples(self,data,max_samples=None):
        self.init(data, max_samples)
        idxs = range(self.data.nnz)
        random.shuffle(idxs)
        self.users, self.items = self.data.nonzero()
        self.users = self.users[idxs]
        self.items = self.items[idxs]
        self.idx = 0
        for _ in xrange(self.num_samples(self.data.nnz)):
            u = self.users[self.idx]
            i = self.items[self.idx]
            j = self.sample_negative_item(self.data[u].indices)
            self.idx += 1
            yield u, i, j
def start(users, items, trainfile, testfile, pu, qi):
    RATE_MATRIX = np.zeros((9238, 7973))
    train_file = trainfile
    train = pd.read_csv(train_file, dtype='str')
    for line in train.values:
        uid = int(float(line[1]))
        iid = int(float(line[2]))
        RATE_MATRIX[uid][iid] = 1
    data = spr.csr_matrix(RATE_MATRIX)
    args = BPRArgs()
    args.learning_rate = 0.05
    num_factors = 50
    model = BPR(num_factors, args, users, items, trainfile, testfile, pu, qi)
    sample_negative_items_empirically = True
    sampler = UniformPairWithoutReplacement(sample_negative_items_empirically)
    num_iters = 500
    model.train(data, sampler, num_iters)

if __name__ == '__main__':
    start()



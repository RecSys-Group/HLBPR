import os
import time
import datetime
from gensim import *
from six import *
import codecs
import numpy as np
from math import exp
import random
from scipy.sparse import coo_matrix
import scipy.sparse as spr
import pandas as pd
import math
from Metrics import BasicMetric


class BPRArgs(object):
    def __init__(self, learning_rate=0.05,
                 bias_regularization=0,
                 user_regularization=0.0025,
                 positive_item_regularization=0.00025,
                 negative_item_regularization=0.00025,
                 update_negative_item_factors=True):
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.update_negative_item_factors = update_negative_item_factors


class BPR(object):
    def __init__(self, D, args, users, items, trainfile, testfile, pu, qi, nn_similar_file, seq_similar_file, users_of_pattern, new_format_seq):
        """initialise BPR matrix factorization model
        D: number of factors
        """
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
        self.nn_simifile = nn_similar_file
        self.seq_simifile = seq_similar_file
        self.users_of_pattern = users_of_pattern
        self.new_format_seq = new_format_seq
        self.initpu = pu
        self.initqi = qi
        self.cxusers = pd.read_csv(self.file_users)
        self.cxitems = pd.read_csv(self.file_items)
        self.cxtrain = pd.read_csv(self.trainfile, dtype='str')
        self.cxtest = pd.read_csv(self.testfile, dtype='str')

    def train(self, data, sampler, num_iters):
        self.init(data)
        old_loss = self.loss_with_three_segments()
        print 'initial loss = {0}'.format(self.loss_with_three_segments())
        for it in xrange(num_iters):
            print 'starting iteration {0}'.format(it)
            '''
            for u, i, j in sampler.generate_samples(self.data):
                self.update_factors(u, i, j)
            '''
            for u, i, k, j in sampler.generate_three_segment_samples(self.data):
                self.update_factors_with_three_segment(u, i, k, j)
            if abs(self.loss_with_three_segments() - old_loss) < 1:
                print 'iteration {0}: loss = {1}'.format(it, self.loss_with_three_segments())
                print 'converge!!'
                break
            else:
                old_loss = self.loss_with_three_segments()
                self.learning_rate *= 0.9
                print 'iteration {0}: loss = {1}'.format(it, self.loss_with_three_segments())
        self.ValidateF1()
        t = pd.DataFrame(self.user_factors)
        t.to_csv('ourbprnnseq50pu')
        t = pd.DataFrame(self.item_factors)
        t.to_csv('ourbprnnseq50qi')
        t = pd.DataFrame(self.item_bias)
        t.to_csv('ourbprnnseq50b')

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
        num_loss_samples = int(100 * self.num_users ** 0.5)
        print 'sampling {0} <user,item i,item k,item j> triples...'.format(num_loss_samples)
        sampler = UniformUserUniformItem(True,self.nn_simifile, self.seq_simifile, self.trainfile, self.users_of_pattern, self.new_format_seq)
        self.loss_samples = [t for t in sampler.generate_three_segment_samples(self.data, num_loss_samples)]
        print 'ok'

    def update_factors_with_three_segment(self, u, i, k, j, w1=1, w2=1, update_u=True, update_i=True, update_k=True):
        """apply SGD update"""
        if k == 10000:
            pass
        else:
            update_j = self.update_negative_item_factors
            x_1 = self.item_bias[i] - self.item_bias[k] \
                  + np.dot(self.user_factors[u, :], self.item_factors[i, :] - self.item_factors[k, :])
            x_2 = self.item_bias[k] - self.item_bias[j] \
                  + np.dot(self.user_factors[u, :], self.item_factors[k, :] - self.item_factors[j, :])
            z_1 = 1.0 / (1.0 + exp(w1 * x_1))
            z_2 = 1.0 / (1.0 + exp(w2 * x_2))
            # update bias terms
            if update_i:
                d = z_1 * w1 - self.bias_regularization * self.item_bias[i]
                self.item_bias[i] += self.learning_rate * d
            if update_k:
                d = z_1 * (-w1) + z_2 * w2 - self.bias_regularization * self.item_bias[k]
                self.item_bias[i] += self.learning_rate * d
            if update_j:
                d = -z_2 * w2 - self.bias_regularization * self.item_bias[j]
                self.item_bias[j] += self.learning_rate * d

            if update_u:
                d = (self.item_factors[i, :] - self.item_factors[k, :]) * w1 * z_1 + (self.item_factors[k,:] - self.item_factors[j,:]) * w2 * z_2 - self.user_regularization * self.user_factors[u,:]
                self.user_factors[u, :] += self.learning_rate * d
            if update_i:
                d = self.user_factors[u, :] * w1 * z_1 - self.positive_item_regularization * self.item_factors[i, :]
                self.item_factors[i, :] += self.learning_rate * d
            if update_k:
                d = -self.user_factors[u, :] * w1 * z_1 + self.user_factors[u,:] * w2 * z_2 - self.positive_item_regularization * self.item_factors[k, :]
                self.item_factors[k, :] += self.learning_rate * d
            if update_j:
                d = -self.user_factors[u, :] * w2 * z_2 - self.negative_item_regularization * self.item_factors[j, :]
                self.item_factors[j, :] += self.learning_rate * d

    def loss_with_three_segments(self):
        ranking_loss = 0
        w1 = 1
        w2 = 1
        for u, i, k, j in self.loss_samples:
            if k == 10000:
                pass
            else:
                x_1 = self.predict(u, i) - self.predict(u, k)
                x_2 = self.predict(u, k) - self.predict(u, j)
                ranking_loss += math.log(1.0 + math.exp(-w1 * x_1))
                ranking_loss += math.log(1.0 + math.exp(-w2 * x_2))

        complexity = 0
        for u, i, k, j in self.loss_samples:
            if k == 10000:
                pass
            else:
                complexity += self.user_regularization * np.dot(self.user_factors[u], self.user_factors[u])
                complexity += self.positive_item_regularization * np.dot(self.item_factors[i], self.item_factors[i])
                complexity += self.negative_item_regularization * np.dot(self.item_factors[k], self.item_factors[k])
                complexity += self.negative_item_regularization * np.dot(self.item_factors[j], self.item_factors[j])

                complexity += self.bias_regularization * self.item_bias[i] ** 2
                complexity += self.bias_regularization * self.item_bias[k] ** 2
                complexity += self.bias_regularization * self.item_bias[j] ** 2
        return ranking_loss + 0.5 * complexity

    def new_get_trunk(self, row):
        for i in range(len(row)):
            if math.isnan(float(row[i])):
                break
        return row[:i]

    def get_trunk(self, goods):
        length = len(goods)
        for i in range(length):
            if goods[i].find('-') > 0:
                break
        return goods[1:i - 1]

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
            # prodeucts = self.get_trunk(row)
            pids = [self.from_product_to_pid(int(p)) for p in prodeucts]
            purchased_list.append(pids)
            uid = self.new_get_trunk(row)[0]
            #uid = self.new_get_trunk(row)[len(prodeucts)+1]
            recom = self.gen_recommendation(int(uid), 5)
            recommendation_list.append(recom)
            print pids, recom
        p, r, f, h = metric.F1_score_Hit_ratio(recommendation_list, purchased_list)
        ndgg = metric.NDGG_k(recommendation_list, purchased_list)
        t = pd.DataFrame([str(f), str(h), str(ndgg)])
        t.to_csv('ourbprnnseq50_result')
        return f

    def predict(self, u, i):
        return self.item_bias[i] + np.dot(self.user_factors[u], self.item_factors[i])


class Sampler(object):
    def __init__(self, sample_negative_items_empirically, nn_similar_file, seq_similar_file, trainfile, users_of_pattern, new_format_seq):
        self.sample_negative_items_empirically = sample_negative_items_empirically
        self.users_of_pattern_file = users_of_pattern
        self.transaction_file = new_format_seq
        self.similar_nn_file = nn_similar_file
        self.similar_seq_file = seq_similar_file
        self.trainfile = trainfile

        self.users_of_pattern = pd.read_csv(self.users_of_pattern_file, dtype='str')
        self.transction = pd.read_csv(self.transaction_file, names=np.array(range(300)))
        self.nn_simi = pd.read_csv(self.similar_nn_file)
        self.seq_simi = pd.read_csv(self.similar_seq_file)
        self.train = pd.read_csv(self.trainfile, dtype='str')


    def init(self, data, max_samples=None):
        self.data = data
        self.num_users, self.num_items = data.shape
        self.max_samples = max_samples

    def get_users_of_product(self, pid):
        uid_list = np.array(self.train.iloc[:, 1])
        pid_list = np.array(self.train.iloc[:, 2])
        u_index = np.argwhere(pid_list == str(float(pid)))
        u_index = [i[0] for i in u_index]
        uids = np.array(self.train.iloc[:, 1])[u_index]
        uids = [int(float(i)) for i in uids]
        return uids

    def get_products_of_user(self, uid):
        uid_list = np.array(self.train.iloc[:, 1])
        pid_list = np.array(self.train.iloc[:, 2])
        p_index = np.argwhere(uid_list == str(float(uid)))
        p_index = [i[0] for i in p_index]
        pids = np.array(self.train.iloc[:, 2])[p_index]
        pids = [int(float(i)) for i in pids]
        return pids

    def get_lasttran_of_user(self, uid):
        line = self.transction.iloc[uid].values
        line_array = np.array(line[0].split(' '))
        split_index = [i[0] for i in np.argwhere(line_array == '-1')]
        if len(split_index) == 1:
            s = 0
        else:
            s = split_index[-2] + 1
        last_tran = line_array[s:split_index[-1]]
        return last_tran

    def gen_users_of_patterns(self, user_lasttran, pid):
        users = []
        # print user_lasttran
        for p in user_lasttran:
            key = str(p) + '@' + str(pid)
            keys = np.array(self.users_of_pattern.iloc[:, 1].values)
            index = np.argwhere(keys == key)
            if len(index) != 0:
                #print pid, p
                user = np.array(self.users_of_pattern.values)[index[0][0]][2]
                users.append(eval(user))
        return users

    def user_item_pref(self, uid, pid):
        uids = self.get_users_of_product(pid)
        dictarray = [dict(eval(i)) for i in np.array(self.nn_simi.iloc[uid])[1:] if isinstance(i, str)]
        neighbor_dict = dict([(i.keys()[0], i.values()[0]) for i in dictarray])
        pref = 0
        for i in uids:
            if neighbor_dict.has_key(str(i)):
                pref += float(neighbor_dict[str(i)])
        return pref

    def user_item_pref_SEQ(self, uid, pid):
        uids = self.get_users_of_product(pid)
        dictarray = [dict(eval(i)) for i in np.array(self.seq_simi.iloc[uid])[1:] if isinstance(i, str)]
        neighbor_dict = dict([(i.keys()[0], i.values()[0]) for i in dictarray])
        pref = 0
        for i in uids:
            if neighbor_dict.has_key(str(i)):
                pref += float(neighbor_dict[str(i)])
        return pref

    def sample_user(self):
        u = self.uniform_user()
        num_items = self.data[u].getnnz()
        assert (num_items > 0 and num_items != self.num_items)
        return u

    def sample_indicate_item(self, items_list):
        j = self.random_item()
        while j not in items_list:
            j = self.random_item()
        return j

    def sample_k_item(self, items_list, uid, score=0):
        j = self.random_item()
        a = 0
        while self.user_item_pref(uid, j) == 0.0 and self.user_item_pref_SEQ(uid, j) == 0.0 and a <= 10:
            j = self.random_item()
            a += 1
        if a > 10:
            return 10000
        else:
            return j

    def sample_j_item(self, items_list, uid, score=0):
        j = self.random_item()
        while j in items_list or self.user_item_pref(uid, j) > 0.0 or self.user_item_pref_SEQ(uid, j) > 0.0:
            j = self.random_item()
        return j

    def sample_negative_item(self, user_items):
        j = self.random_item()
        while j in user_items:
            j = self.random_item()
        return j

    def uniform_user(self):
        return random.randint(0, self.num_users - 1)

    def random_item(self):
        """sample an item uniformly or from the empirical distribution
           observed in the training data
        """
        if self.sample_negative_items_empirically:
            # just pick something someone rated!
            u = self.uniform_user()
            i = random.choice(self.data[u].indices)
        else:
            i = random.randint(0, self.num_items - 1)
        return i

    def num_samples(self, n):
        if self.max_samples is None:
            return n
        return min(n, self.max_samples)


class UniformUserUniformItem(Sampler):
    def get_user_purchased(self, data, uid):
        index = np.argwhere(np.array(data.iloc[:, 1]) == int(uid))
        index_p = [i[0] for i in index]
        products = list(np.array(data.iloc[:, 2])[index_p])
        result = [int(float(i)) for i in products]
        return result

    def generate_three_segment_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        self.users, self.items = self.data.nonzero()
        a = 0
        for _ in xrange(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            i = random.choice(self.data[u].indices)
            k = self.sample_k_item(self.data[u].indices, u)
            j = self.sample_j_item(self.data[u].indices, u)
            if k in self.data[u].indices:
                k = 10000
            # print a
            a += 1
            yield u, i, k, j



class UniformPairWithoutReplacement(Sampler):
    def get_user_purchased(self, data, uid):
        index = np.argwhere(np.array(data.iloc[:, 1]) == int(uid))
        index_p = [i[0] for i in index]
        products = list(np.array(data.iloc[:, 2])[index_p])
        result = [int(float(i)) for i in products]
        return result

    def generate_three_segment_samples(self, data, max_samples=None):
        self.init(data, max_samples)
        idxs = range(self.data.nnz)
        random.shuffle(idxs)
        self.users, self.items = self.data.nonzero()
        self.users = self.users[idxs]
        self.items = self.items[idxs]
        self.idx = 0
        for _ in xrange(10000):
            u = self.users[self.idx]
            i = self.items[self.idx]
            k = self.sample_k_item(self.data[u].indices, u)
            j = self.sample_j_item(self.data[u].indices, u)
            if k in self.data[u].indices:
                k = 10000
            self.idx += 1
            # print self.idx
            yield u, i, k, j


def gen_input_data(trainfile):
    RATE_MATRIX = np.zeros((9238, 7973))
    train_file = trainfile
    train = pd.read_csv(train_file, dtype='str')
    for line in train.values:
        uid = int(float(line[1]))
        iid = int(float(line[2]))
        RATE_MATRIX[uid][iid] = 1
    data = spr.csr_matrix(RATE_MATRIX)
    return data


def test_model(data, users, items, trainfile, testfile, pu, qi, nn_similar_file, seq_similar_file, users_of_pattern, new_format_seq, learning_rate, b_regularization=1, u_regularization=0.01, p_regularization=0.01,
               n_regularization=0.01, factor=50, iter=100):
    args = BPRArgs()
    args.learning_rate = learning_rate
    args.bias_regularization = b_regularization
    args.user_regularization = u_regularization
    args.positive_item_regularization = p_regularization
    args.negative_item_regularization = n_regularization
    num_factors = factor
    model = BPR(num_factors, args, users, items, trainfile, testfile, pu, qi, nn_similar_file, seq_similar_file, users_of_pattern, new_format_seq)
    sample_negative_items_empirically = True
    sampler = UniformPairWithoutReplacement(sample_negative_items_empirically, nn_similar_file, seq_similar_file, trainfile, users_of_pattern, new_format_seq)
    num_iters = iter
    model.train(data, sampler, num_iters)
def start(users, items, trainfile, testfile, pu, qi, nn_similar_file, seq_similar_file, users_of_pattern, new_format_seq):
    data = gen_input_data(trainfile)
    test_model(data, users, items, trainfile, testfile, pu, qi, nn_similar_file, seq_similar_file, users_of_pattern, new_format_seq, 0.005, 0.0, 0.0003, 0.00025, 0.00025)

if __name__ == '__main__':
    start()







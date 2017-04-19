#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import json
import cPickle as pickle

now_path = os.getcwd() + '/'
data_path = now_path + '../dataset/'
business_path = data_path + 'businesses.json'
user_path = data_path + 'users.json'
star_path = data_path + 'stars.pk'
libsvm_path = data_path + 'libsvm_train.txt'

# load data from files
with open(user_path, 'r') as f:
    users = json.load(f)
with open(business_path, 'r') as f:
    businesses = json.load(f)
stars = pickle.load(open(star_path, 'rb'))

'''
------------------------------------------------
Features and index:
users_info(19 -> 1-19) = \
'user_id':
{
  "loc": [33.3055295, -111.9027999], "votes": [441, 202, 372], "loc_num": 1,
  "avg_star": 3.69, "cates": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], "rev_num": 314
}
business_info(23 -> 20-42) = \
'business_id':
{
  'loc': [33.6371076, -112.2249764], 'votes': [4.857617605789824, 4.034214786832164, 5.846698309601132],
  'avg_star': 3.5, 'cates': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'open'(don't use): 1,
  'rev_num': 6, 'ckins': [2, 6, 0, 1, 2]
}
------------------------------------------------
Label:
{('business_id', 'user_id'): star}
------------------------------------------------
'''

f = open(libsvm_path, 'w')
num_now = 0
for (bid, uid) in stars:
    num_now += 1
    if num_now % 10000 == 0:
        print 'now: %d * 10000' % (num_now / 100000)
    line = ''
    if bid in businesses and uid in users:
        # label
        line += str(stars[(bid, uid)])
        index = 1
        # features - user
        for i in xrange(2):
            line = line + ' ' + str(index) + ':' + str(users[uid]['loc'][i])
            index += 1
        for i in xrange(3):
            if users[uid]['votes'][i] != 0:
                line = line + ' ' + str(index) + ':' + str(users[uid]['votes'][i])
            index += 1
        line = line + ' ' + str(index) + ':' + str(users[uid]['loc_num'])
        index += 1
        line = line + ' ' + str(index) + ':' + str(users[uid]['avg_star'])
        index += 1
        for i in xrange(11):
            if users[uid]['cates'][i] != 0:
                line = line + ' ' + str(index) + ':' + str(users[uid]['cates'][i])
            index += 1
        line = line + ' ' + str(index) + ':' + str(users[uid]['rev_num'])
        index += 1
        # features - business
        for i in xrange(2):
            line = line + ' ' + str(index) + ':' + str(users[uid]['loc'][i])
            index += 1
        for i in xrange(3):
            if users[uid]['votes'][i] - 0.0 > 0.00001:
                line = line + ' ' + str(index) + ':' + str(users[uid]['votes'][i])
            index += 1
        line = line + ' ' + str(index) + ':' + str(users[uid]['avg_star'])
        index += 1
        for i in xrange(11):
            if users[uid]['cates'][i] != 0:
                line = line + ' ' + str(index) + ':' + str(users[uid]['cates'][i])
            index += 1
        line = line + ' ' + str(index) + ':' + str(users[uid]['rev_num'])
        index += 1
        for i in xrange(5):
            if users[uid]['cates'][i] != 0:
                line = line + ' ' + str(index) + ':' + str(users[uid]['cates'][i])
            index += 1
        line += '\n'
    f.write(line)
print 'There are %d records' % num_now
f.close()

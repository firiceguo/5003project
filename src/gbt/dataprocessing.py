#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import json

now_path = os.getcwd() + '/'
yelp_path = now_path + '/yelp/'
business_path = yelp_path + 'business.json'
checkin_path = yelp_path + 'checkin.json'
review_path = yelp_path + 'review.json'
user_path = yelp_path + 'user.json'
data_path = now_path + '/yelp/big_dataset/'

'''
!!! Don't use the following templates directly !!!
=======================================================================================================================
user_template = \
    {'loc': [], 'loc_num': 0, 'rev_num': 0, 'cates': [0]*10, 'avg_star': 0, 'votes': []}
{
  "loc": [33.3055295, -111.9027999], "votes": [441, 202, 372], "loc_num": 1,
  "avg_star": 3.69, "cates": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], "rev_num": 314
}

business_template = \
    {'loc': [], 'rev_num': 0, 'cates': [], 'avg_star': 0, 'votes': [], 'ckins': [], 'open': 1}
{
  'loc': [33.6371076, -112.2249764], 'votes': [4.857617605789824, 4.034214786832164, 5.846698309601132],
  'avg_star': 3.5, 'cates': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'open': 1, 'rev_num': 6, 'ckins': [2, 6, 0, 1, 2]
}

review_template = \
    {'star': 0, 'votes': []}
'''
categories = {}


def reduceDimension(dic, num=10):
    assert len(
        dic) >= num, 'There are not so much categories! (len(categories)>=cate_num)'
    new = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    return [new[i][0] for i in xrange(num)]


def genJsonDataset(paths):
    global categories
    cate_num = 10

    # users = {key=user_id: value=user_template}
    # Yes: #review, average star, votes
    # No : location, #location, categories
    users = {}
    # with open(paths[0], 'r') as f:
    #     info = json.load(f)
    user_file = open(paths[0], 'r')
    item = user_file.readline()
    while item:
        info = json.loads(item)
        if info['user_id'] not in users:
            uid = info['user_id']
            users[uid] = {'loc': [], 'loc_num': 0, 'rev_num': 0,
                          'cates': [0] * 10, 'avg_star': 0, 'votes': []}
            users[uid]['rev_num'] = info['review_count']
            users[uid]['avg_star'] = info['average_stars']
            users[uid]['votes'] = \
                [info['useful'], info['funny'], info['cool']]
        item = user_file.readline()
    user_file.close()
    print 'load dict users done!'
    print 'users = {key=user_id: value=user_template} * %d' % len(users)
    print users[info['user_id']]
    print

    # businesses = {key=business_id: value=business_template}
    # Yes: location, #review, average star, open, categories(origin)
    # No : categories(set-up), votes, check-ins
    # generate categories
    businesses = {}
    business_file = open(paths[3], 'r')
    item = business_file.readline()
    while item:
        info = json.loads(item)
        if info['business_id'] not in businesses:
            businesses[info['business_id']] = \
                {'loc': [], 'rev_num': 0, 'cates': [], 'avg_star': 0,
                    'votes': [], 'ckins': [], 'open': 1}
            businesses[info['business_id']]['rev_num'] = info['review_count']
            businesses[info['business_id']]['avg_star'] = info['stars']
            businesses[info['business_id']]['loc'] = [
                info['latitude'], info['longitude']]
            businesses[info['business_id']]['cates'] = info['categories']
            if not info['is_open']:
                businesses[info['business_id']]['open'] = 0
        if info['categories'] is not None:
            for cate in info['categories']:
                try:
                    categories[cate] += 1
                except:
                    categories[cate] = 1
        item = business_file.readline()
    business_file.close()
    print 'load dict businesses done!'
    print 'businesses = {key=business_id: value=business_template} * %d' % len(businesses)
    print businesses[info['business_id']]
    print

    # reviews = {key=business_id: value={user_id: review_template}}
    # Yes: star, votes
    # No : -
    # stars = {key=(business_id, user_id), value={'stars': stars, 'rev_id':
    # review_id}}
    reviews = {}
    stars = {}
    review_file = open(paths[1], 'r')
    item = review_file.readline()
    while item:
        info = json.loads(item)
        stars[(info['business_id'], info['user_id'])] = {
            'stars': info['stars'], 'rev_id': info['review_id']}
        if info['business_id'] not in reviews:
            reviews[info['business_id']] = {
                info['user_id']: {'star': 0, 'votes': []}}
            reviews[info['business_id']][info['user_id']
                                         ]['star'] = info['stars']
            reviews[info['business_id']][info['user_id']]['votes'] = \
                [info['useful'], info['funny'], info['cool']]
        elif info['user_id'] not in reviews[info['business_id']]:
            reviews[info['business_id']][info['user_id']] = {
                'star': 0, 'votes': []}
            reviews[info['business_id']][info['user_id']
                                         ]['star'] = info['stars']
            reviews[info['business_id']][info['user_id']]['votes'] = \
                [info['useful'], info['funny'], info['cool']]
        item = review_file.readline()
    review_file.close()
    '''
    with open(data_path + 'reviews.json', 'w') as f:
        for key in reviews:
            review = reviews[key]
            review['b_id'] = key[0]
            review['u_id'] = key[1]
            j = json.dumps(review)
            f.write(j)
            f.write('\n')
    # pickle.dump(stars, open(data_path + 'stars.pk', 'wb'))
    print 'load & dump dict reviews & stars done!'
    # print 'reviews = {key=business_id: value={user_id: review_template}} * %d' % len(reviews)
    # print reviews[info['business_id']]
    # print stars[(info['business_id'], info['user_id'])]
    print
    '''
    # configure categories
    # configure location and #location for users
    important_cate = reduceDimension(categories, cate_num)
    print 'important categories:'
    print important_cate
    print
    for bid in businesses:
        local_cate = businesses[bid]['cates']
        # configure business categories
        businesses[bid]['cates'] = [0] * 10
        num = 0
        for i in xrange(cate_num):
            if local_cate is not None:
                if important_cate[i] in local_cate:
                    businesses[bid]['cates'][i] = 1
                    num += 1
        businesses[bid]['cates'].append(num)
        # configure user info
        if bid in reviews:
            loc = businesses[bid]['loc']
            if len(loc) != 2:
                continue
            for uid in reviews[bid]:
                if uid in users:
                    # location
                    users[uid]['loc'].append(loc)
                    # categories
                    for i in xrange(cate_num):
                        users[uid]['cates'][i] = users[uid]['cates'][i] + \
                            businesses[bid]['cates'][i]
    for uid in users:
        users[uid]['loc_num'] = len(users[uid]['loc'])
        users[uid]['loc'] = [sum(users[uid]['loc'][i][0] for i in xrange(users[uid]['loc_num'])) /
                             users[uid]['loc_num'],
                             sum(users[uid]['loc'][i][1] for i in xrange(users[uid]['loc_num'])) /
                             users[uid]['loc_num']]
        num = 0
        for x in users[uid]['cates']:
            if x > 0:
                num += 1
        users[uid]['cates'].append(num)
    print 'configure categories and location(user) done!'
    with open(data_path + 'gbt_users.json', 'w') as f:
        for key in users:
            user = users[key]
            user['u_id'] = key
            j = json.dumps(user)
            f.write(j)
            f.write('\n')
    print 'dump users.json done!\n'

    # configure business_votes = sum(weights[i] * votes[i]),
    # where weights[i] = vote_of_user_who_own_vote[i]/sum(votes)
    for bid in reviews:
        weights = []
        votes = []
        sum_vote = [0, 0, 0]
        for uid in reviews[bid]:
            if uid not in users:
                continue
            sum_vote = [sum_vote[i] + users[uid]['votes'][i]
                        for i in xrange(len(sum_vote))]
            votes.append(reviews[bid][uid]['votes'])
        for uid in reviews[bid]:
            if uid not in users:
                continue
            temp = []
            for i in xrange(3):
                if sum_vote[i] == 0:
                    temp.append(0.0)
                else:
                    temp.append(users[uid]['votes'][i] / float(sum_vote[i]))
            weights.append(temp)
        businesses[bid]['votes'] = \
            [sum(weights[i][j] * votes[i][j]
                 for i in xrange(len(weights))) for j in xrange(3)]
    print 'configure votes for businesses done!'
    print businesses[bid]
    print

    # configure check-in data according to 'vsu_RecSys2013' section 2.4
    # days: 3 categories: (Sat, Sun)->1, (Mon, Tue)->2, (Wed, Thu, Fri)->3
    # hour: 5 categories: [0:5]->1, [6:9]->2, [10:15]->3, [16:19]->4, [20:23]->5
    # Cartesian_product(days, hours), primary_key=days, secondary_key=hour
    # checkin = {key=business_id, value=[15 values from Cartesian product]}
    checkin_file = open(checkin_path, 'r')
    sum_checkin = {}
    item = checkin_file.readline()
    dayMap = {'Sun': 0, 'Mon': 1, 'Tue': 2,
              'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6}
    while item:
        info = json.loads(item)
        try:
            businesses[info['business_id']]['ckins'] = [0] * 15
        except:
            item = checkin_file.readline()
            continue
        for time in info['time']:
            day = dayMap[time.split('-')[0]]
            hour = int(time.split('-')[1].split(':')[0])
            num_ckin = int(time.split(':')[1])
            if day in [0, 6]:
                primary_key = 0
            elif day in [1, 2]:
                primary_key = 1
            else:
                primary_key = 2
            if hour in [0, 1, 2, 3, 4, 5]:
                secondary_key = 0
            elif hour in [6, 7, 8, 9]:
                secondary_key = 1
            elif hour in [10, 11, 12, 13, 14, 15]:
                secondary_key = 2
            elif hour in [16, 17, 18, 19]:
                secondary_key = 3
            else:
                secondary_key = 4
            index = primary_key * 5 + secondary_key
            businesses[info['business_id']]['ckins'][index] += num_ckin
            try:
                sum_checkin[index] += 1
            except:
                sum_checkin[index] = 1
        item = checkin_file.readline()
    checkin_file.close()
    print 'load checkin data done!'
    print businesses[bid]
    print
    # reduce dimension from 15 to 5
    important_checkin = reduceDimension(sum_checkin, num=5)
    print 'important check-ins:'
    print important_checkin
    print
    for bid in businesses:
        new_checkin = []
        for i in xrange(len(important_checkin)):
            try:
                new_checkin.append(
                    businesses[bid]['ckins'][important_checkin[i]])
            except:
                new_checkin.append(0)
        businesses[bid]['ckins'] = new_checkin
    print 'Dimensional reduction done! (from 15 to 5)'
    with open(data_path + 'gbt_businesses.json', 'w') as f:
        for key in businesses:
            business = businesses[key]
            business['b_id'] = key
            j = json.dumps(business)
            f.write(j)
            f.write('\n')
    print 'Dump businesses data done!'
    print businesses[bid]
    print


if __name__ == '__main__':
    now_path = os.getcwd() + '/'
    yelp_path = now_path + '../yelp/'
    business_path = yelp_path + 'business.json'
    checkin_path = yelp_path + 'checkin.json'
    review_path = yelp_path + 'review.json'
    user_path = yelp_path + 'user.json'
    data_path = now_path + '/yelp/big_dataset/'
    genJsonDataset([user_path, review_path, checkin_path, business_path])

# 说明

### 目录结构

所有的数据均不存在，只留下了空文件用来表示结构


```
└─sguo
    │  README.md
    │
    ├─dataset
    │      businesses.json
    │      reviews.json
    │      stars.pk
    │      users.json
    │
    ├─src
    │      getdata-output.txt
    │      getdata.py
    │
    └─yelp_training_set
            yelp_training_set_business.json
            yelp_training_set_checkin.json
            yelp_training_set_review.json
            yelp_training_set_user.json
```

### 代码

在`./src` 目录中

- `getdata.py`

	用来生成`dataset`目录中的 json 和 pickle 数据。由于json不支持`(user_id, business_id)`这样的作为key，因此直接用pickle进行序列化。

### 思路

Reference：[vsu_RecSys2013.pdf](https://github.com/firiceguo/Recommendation-NLP/blob/master/reference/zhangrong/vsu_RecSys2013.pdf) 的 abstract 和第二章

1. 处理`yelp_training_set_user.json`，构建名为`users`的字典；
	
2. 处理`yelp_training_set_business.json`，构建名为`businesses`的字典，同时构建所有的`categories`，为接下来*计算user的categories*和*降维*做准备；

3. 处理`yelp_training_set_review.json`，构建名为`reviews`和`stars`的字典；

4. 用函数`reduceDimension(dic, num)`对所有`categories`进行降维处理，计算每个user的categories，同时处理所有user的`location信息`；

5. 对所有的business计算其对应的votes，就是加权计算，分两步：

	每个user的权值乘每个user的投票：`business_votes = sum(weights[i] * votes[i])`

	权值等于投票的这个user的情况除以所有投给这个business的user的情况：`weights[i] = vote_of_user_who_own_vote[i] / sum(votes)`

6. 处理`yelp_training_set_checkin.json`，统计每家店的checkin信息，并用函数`reduceDimension(dic, num)`对所有`checkin`进行降维处理，方法参考**Reference - section 2.4**。

数据例子参考代码输出日志 `getdata-output.log`。

### python 的坑（debug过程）

1. 笔误：`cates` 写成 `cate` 等等。

2. **重要**：python参数引用问题，本来想简单一点用`template`来写，但是碰见python的参数引用问题出错。简要重现如下：

	```
	>>> template = {'a': 1, 'b':2}
	>>> keys = ['x', 'y', 'z']
	>>> dic1 = {}
	>>> for key in keys:
	...   if key not in dic1:
	...     dic1[key] = template
	...     dic1[key]['a'] = key
	...     print dic1			# 输出字典 dic1
	...     print id(dic1[key]) # 输出地址
	...
	{'x': {'a': 'x', 'b': 2}}
	37581688
	{'y': {'a': 'y', 'b': 2}, 'x': {'a': 'y', 'b': 2}}
	37581688
	{'y': {'a': 'z', 'b': 2}, 'x': {'a': 'z', 'b': 2}, 'z': {'a': 'z', 'b': 2}}
	37581688
	```

	结果导致左右的value都是一样的，原因是赋值的时候是传引用，基本数据类型赋值其实是重新构造并指向了一个新的对象，所以本质上依然是引用传递。

3. 对于除法，注意处理**除数为 0 的情况**以及**int和float的转化**。

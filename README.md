
#### 介绍

[瑞金医院MMC人工智能辅助构建知识图谱大赛-第一赛季](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.55d733afYIca7m&raceId=231687)

#### 数据

从给定数据中抽出一份作为测试样本，对余下样本按照train : dev = 10 : 1的比例划分

#### 模型

BiLSTM+CRF

#### 实现

Linux+Python3.6+Keras2.2.4+Tensorflow1.11.0 

#### 结果

线上分数=0.71


#### 思考

赛题第一个阶段建模为一个命名实体识别任务(NER)，但是还没思考好第二阶段的方法。通过调研发现，实体识别和关系抽取两个子任务可以通过一个模型实现，比起前者，后者似乎更有趣一些。







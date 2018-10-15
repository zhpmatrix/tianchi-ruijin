
#### 介绍

[瑞金医院MMC人工智能辅助构建知识图谱大赛-第一赛季](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.55d733afYIca7m&raceId=231687)

#### 数据

这个Baseline是基于第一批放出的数据完成的，只有四份文件。

#### 模型

BiLSTM+CRF

#### 实现

Linux+Python3.6+Keras2.2.4+Tensorflow1.11.0 

#### 结果

| optimizer | word embed dim | hidden units | batch size | best epoch | dev acc | test acc（micro/macro,weighted） |
| :------| ------: | ------: |-----:|----:|------:|:-------:|
| Adam | 200 | 200 |128|73|0.84116|0.94/0.63/0.94）|

_提示：数据较少，基本没有参考意义_





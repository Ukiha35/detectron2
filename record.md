# CREMI62

### MODEL TRAINING PARAMETERS

##### 3d_fullres fold 0 【default settings】

patch size: [24, 256, 256]

resolution: [40,4,4]

batch_size: 2

epoch: 1000

##### 3d_fullres fold 2 

patch size: [12, 128, 128]

resolution: [40,4,4]

batch_size: 4

epoch: 1000

##### 3d_lowres fold 2 【default settings】

patch size: [32,160,160]

resolution: [40,15,15]

batch_size: 2

epoch:1000

### MODEL INFERENCE PARAMETERS

##### one stage baseline

model: 3d_fullres fold 0

step: 0.5

patch size: [24,256,256]

resolution: [40,4,4]

##### stage one in two stage baseline

model: 3d_lowres fold 2

step: 0.5

patch size: [32,160,160]

resolution: [40,20,20]

##### heatmap to bounding box

axis_thr: 4.0

area_thr: 6.0

roi_threshold: 0.001

prior_downsample: [3.0,16.0,16.0]

patch_size: [12,128,128] 

canvas_size: [24,256,256]

nms_threshold: 0.75

nmm_threshold: 0.35

feature extracting network: 3d_fullres fold 2

downsample_mode: center

pooling_mode: max

nmm_recrop: [12,128,128]

feature dim: 1287

##### assignnet config

lambda: 0.03

checkpoint epoch: 913

checkpoint epoch2: 1003 with nms threshold: 0.65

start label: False

global info: False

feature extractor: fold 2

##### stage two in two stage baseline

model: 3d_fullres fold 2

postprocess: crop and resize

crop_threshold: [12,128,128]

use_prior: False

### results record

|                    | 一阶段耗时 | 二阶段耗时 |   mAP    | size                                                 |
| :----------------: | :--------: | :--------: | :------: | ---------------------------------------------------- |
|     一阶段base     |   17.60s   |     /      | 46.72153 | 250*250                                              |
| 二阶段base(全父类) |   0.40s    |   13.50s   | 47.64002 | max: ~550\*550<br />mean: ~230\*230<br />min:~50\*50 |
| 二阶段base(全子类) |   0.40s    |   48.00s   | 45.04325 | max: ~180\*180<br />mean: ~100\*100<br />min:~30\*30 |
|    最终加速结果    |   0.40s    |            |          |                                                      |



# CREMI63 FAFB

### MODEL TRAINING PARAMETERS

##### 3d_fullres fold 3 【default settings】

用所有体积来训练

patch size: [24, 256, 256]

resolution: [40,4,4]

batch_size: 2

epoch: 1000

##### 3d_fullres fold 4

用所有体积来训练

patch size: [12, 128, 128]

resolution: [40,4,4]

batch_size: 4

epoch: 1000

##### 3d_lowres fold 4

用所有体积来训练

patch size: [80,160,160]

resolution: [40,15,15]

batch_size: 2

epoch:1000

### MODEL INFERENCE PARAMETERS

##### one stage baseline

model: 3d_fullres fold 3

step: 0.5

patch size: [24,256,256]

resolution: [40,4,4]

##### stage one in two stage baseline

model: 3d_lowres fold 4

step: 0.5

patch size: [80,160,160]

resolution: [40,20,20]

##### heatmap to bounding box

axis_thr: 4.0

area_thr: 6.0

roi_threshold: 0.001

prior_downsample: [3.0,16.0,16.0]

patch_size: [12,128,128] 

canvas_size: [24,256,256]

nms_threshold: 0.75

nmm_threshold: 0.35

feature extracting network: 3d_fullres fold 2

downsample_mode: center

pooling_mode: max

nmm_recrop: [12,128,128]

feature dim: 1287

##### assignnet config

lambda: 0.03

checkpoint epoch: 913

start label: False

global info: False

feature extractor: fold 2

##### stage two in two stage baseline

model: 3d_fullres fold 2

postprocess: crop and resize

crop_threshold: [12,128,128]

use_prior: False

### results record

**part 1**

|                   | 一阶段耗时 | 二阶段耗时 |   IOU   |
| :---------------: | :--------: | :--------: | :-----: |
|    一阶段base     |  3602.08s  |     /      | 0.96020 |
| 小patch一阶段base |  5789.66s  |     /      | 0.97302 |
|    二阶段base     |   59.54s   |  2895.19s  | 0.94521 |
|   最终加速结果    |   59.54s   |  516.22s   | 0.92912 |

**part 2**

|                   | 一阶段耗时 | 二阶段耗时 |   IOU   |
| :---------------: | :--------: | :--------: | :-----: |
|    一阶段base     |  3600.98s  |     /      | 0.90700 |
| 小patch一阶段base |  5778.76s  |     /      | 0.91332 |
|    二阶段base     |   59.47s   |  3310.59s  | 0.87647 |
|   最终加速结果    |   59.47s   |  601.68s   | 0.84896 |

**part 3**

|                   | 一阶段耗时 | 二阶段耗时 |   IOU   |
| :---------------: | :--------: | :--------: | :-----: |
|    一阶段base     |  3606.46s  |     /      | 0.80671 |
| 小patch一阶段base |  5970.93s  |     /      | 0.85515 |
|    二阶段base     |   59.56s   |  1262.71s  | 0.74577 |
|   最终加速结果    |   59.56s   |  249.49s   | 0.67876 |

**part 4**

|                   | 一阶段耗时 | 二阶段耗时 |   IOU   |
| :---------------: | :--------: | :--------: | :-----: |
|    一阶段base     |  3601.62s  |     /      | 0.79484 |
| 小patch一阶段base |  5731.65s  |     /      | 0.88728 |
|    二阶段base     |   59.51s   |  1843.55s  | 0.83424 |
|   最终加速结果    |   59.51s   |  337.75s   | 0.79700 |

**part 1 full**

|                   | 一阶段耗时 | 二阶段耗时 |   IOU   |
| :---------------: | :--------: | :--------: | :-----: |
|    一阶段base     |   46.02h   |     /      | 0.95239 |
| 小patch一阶段base |            |     /      |         |
|    二阶段base     |   0.80h    |   30.74h   | 0.93593 |
|   最终加速结果    |   0.80h    |   5.585h   | 0.91744 |





# COVID 101

### MODEL TRAINING PARAMETERS

##### 3d_fullres fold 0  【default settings】

patch size: [56, 192, 192]

resolution: [1.5, 0.6835939884185791, 0.6835939884185791]

batch_size: 2

epoch: 1000

##### 3d_fullres fold 2

patch size: [56, 96, 96]

resolution: [1.5, 0.6835939884185791, 0.6835939884185791]

batch_size: 2

epoch: 1000

##### 3d_lowres fold 0 【default settings】:

patch size: [64,192,192]

resolution: [2.709166852004121, 1.3896058308690493, 1.3896058308690493]

batch_size: 8

epoch:1000

### MODEL INFERENCE PARAMETERS

##### one stage baseline

model: 3d_fullres fold 0

step: 0.5

patch size: [56,192,192]

resolution: [1.5, 0.6835939884185791, 0.6835939884185791]

##### stage one in two stage baseline

model: 3d_lowres fold 2

step: 0.5

patch size: [64,96,96]

resolution: [2.709166852004121, 1.3896058308690493, 1.3896058308690493]

disable test time augmentation!

use body edge crop!

##### heatmap to bounding box

axis_thr: 0.0

area_thr: 0.0

roi_threshold: 0.5

prior_downsample: [10.0,16.0,16.0]

patch_size: [56,96,96] 

canvas_size: [56,192,192]

nms_threshold: 0.7

nmm_threshold: 0.2

feature extracting network: 3d_fullres fold 2 

downsample_mode: center

pooling_mode: max

nmm_recrop: [56,96,96]

feature dim: 1287

##### assignnet config

lambda: 0.05

checkpoint epoch: 82

start label: False

global info: False

feature extractor: fold 2

##### stage two in two stage baseline

model: 3d_fullres fold 2

postprocess: crop and resize

crop_threshold: [56,96,96]

use_prior: False

### results record

|                                   | 一阶段耗时 | 二阶段耗时 |  dice   | surface dice |
| :-------------------------------: | :--------: | :--------: | :-----: | :----------: |
|            一阶段base             |  2167.76s  |     /      | 0.62618 |   0.71777    |
|         小patch一阶段base         |  2390.75s  |     /      | 0.58996 |   0.66862    |
|            二阶段base             |   40.81s   |  864.83s   | 0.63225 |   0.72419    |
|   二阶段全父类框（resize only）   |   40.81s   |   79.23s   | 0.62263 |   0.71010    |
| 二阶段全父类框（crop and resize） |   40.81s   |   80.28s   | 0.63008 |   0.72168    |
|           最终加速结果            |   40.81s   |  174.15s   | 0.64992 |   0.73184    |
|       加速结果（取消拼图）        |   40.81s   |  171.01s   | 0.62975 |   0.72263    |



# WORD 100 【not done yet】

### MODEL TRAINING PARAMETERS

##### 3d_fullres fold 0  【default settings】

patch size: [64, 192, 160]

resolution: [3.0, 0.9765625, 0.9765625]

epoch: 800

disable data mirroring augmentation!

##### 3d_fullres fold 2

patch size: [64, 64, 64]

resolution: [3.0, 0.9765625, 0.9765625]

epoch: 1000

disable data mirroring augmentation!

##### 3d_lowres fold 2:

patch size: [96,128,128]

resolution: [6,4,4]

epoch:1000

disable data mirroring augmentation!

### MODEL INFERENCE PARAMETERS

##### one stage baseline

model: 3d_fullres fold 0

step: 0.5

patch size: [64, 192, 160]

resolution: [3.0, 0.9765625, 0.9765625]

disable test time augmentation!

##### stage one in two stage baseline

model: 3d_lowres fold 2

step: 0.5

patch size: [96,128,128]

resolution: [6,4,4]

disable test time augmentation!

##### heatmap to bounding box 【not done yet】

axis_thr: 0.0

area_thr: 0.0

roi_threshold: 0.9

prior_downsample: [3.0,3.0,3.0]

patch_size: [64,64,64] 

canvas_size: [288,64,64]

nms_threshold: 0.55

nmm_threshold: 0.5

feature extracting network: 3d_fullres fold 2

downsample_mode: center

pooling_mode: mean

nmm_recrop: [64,74,74]

feature dim: 1287

##### assignnet config【not done yet】

lambda: 0.1275

checkpoint epoch: 48

start label: False

global info: False

##### stage two in two stage baseline【not done yet】

model: 3d_fullres fold 2

postprocess: crop and resize

crop_threshold: [64,74,74]

use_prior: False

### results record:

|                                   | 一阶段耗时 | 二阶段耗时 |  dice   |  hd95   |
| :-------------------------------: | :--------: | :--------: | :-----: | :-----: |
|            一阶段base             |   51.31s   |     /      | 0.87576 | 5.15511 |
|            二阶段base             |   0.65s    |   22.80s   | 0.87987 | 4.87379 |
|   二阶段全父类框（resize only）   |   0.65s    |            |         |         |
| 二阶段全父类框（crop and resize） |   0.65s    |   8.88s    | 0.87695 | 5.00620 |
|           最终加速结果            |   0.65s    |   5.19s    | 0.87669 | 5.61314 |
|       加速结果（取消拼图）        |   0.65s    |            |         |         |





nms阈值，nmm阈值，



一阶段base还要保存时间，跑一阶段base

想想怎么训lowres，开始训，重开一个yaml

2stage代码写





修改一下粗检测器的loss，修改数据集切分方式













第一阶段筛选以损失性能的情况下提升了速度；第二阶段排列和拼接在不损失速度（不添加无用的背景上下文）的情况下加入了上下文信息，提升了性能。

筛选和排列基于强化学习，拼接基于两种启发式算法。



实验：

1. 筛选：全父类/全子类
2. 拼接：canvas大小(一个canvas塞1、4、8个...)
3. 排列：随机排/不随机排 统计图表(1和2和3一起做)
4. 不同放缩比例(crop,resize)(三个维度一起做曲线，同时加上不同nmm的平均大小)
5. 拼接方法：同大小/同放缩比例
6. 在fafb上测试 单独放，时间对比
7. 在cremi上不同训练集/验证集分割方式加速，5和6和7一起做表格
8. roi_threshold和召回率要画图。
9. 不同patch，nms_threshold，step_size大小？可视化出来一块间隙的大小对应patch大小合适际可，nms略大于step_size，step_size默认。
10. 强化学习出来的排序insight，可视化。
11. 分割器模型参数画图？



提取 筛选 排序 拼接



长度

比例，前两章篇幅

introduction内容，摘要。针对问题，工作，提出了什么的方法首先，，其次，，，可以长。



1.1 重要性，配图

1.2 速度太慢，因此加速。引出问题。2*（当前虽然啥方法，但仍然有问题），一个问题对应一个部分。配图

目前已经有什么方法，但是还是不行。

1.3 大的摘要。针对。。我们。。 配图

1.4 章节安排，整体框图



第二章

细节讨论问题。别人的方法。细节：每一类问题，其中代表性的是。。。

可以介绍传统医学检测，和加速无关。

后面**几个问题就是几块，一一对应**



二阶段第一章要写

上下文割裂要具体写

1.3需要从图上看到 通过什么方法做到加速 要讲清楚 排列组合

1.3.1 先讲思路 先找出candidate 每个画图两个问题

怎么选选的太多选的太少，因此要树

画图不能绕弯，横过来

两个问题三个章节怎么对应

参考文献图！

加速方法没讨论

2.2专门加速要写





如图所树 不用搜索整个图像，同时为上下文奠定基础。。。

要承接

如上图所说



框图局部放大

要画框图、



3-4



每一章画框图 每一张干啥，怎么衔接



自己的东西讲清楚

干活





每一张都要提加速



把立方体变成一张图





要让人看懂

要有干货 要讨论 好处



实验要多 8-10页







前面框图再补

图连续，长 

canvas拼法接在里面



修改预处理图

表要加粗 本文方法 ablation分开

unet特征提取起名改

拆分表





















血动力学 高精度相机 多普勒血流



"在电镜、ct这种一类的数据集中各选了一个作为验证“



韧性强？



怎么人工和深度学习结合？



为什么nnunet和vnet差不多？



为什么我们的方法更好？



多个类型数据集对于reviewer的话也不一定都熟悉，3个不太够，4-5个比较合适，也不需要更多了





fafb上可视化上是否可以对比官方的预测结果？

















The paper proposes an unsupervised underwater image enhancement(UIE) method. Firstly, a Fast Language-Image Pretraining(FLIP) model is trained to capture the prompt embeddings paired with high-quality and low-quality images. After that, the underwater dark channel prior (DCP) is used to obtain pre-restored images. Finally a U-net enhance network guided by the FLIP model is employed to process the pre-restored images, and produces the enhanced images. Plenty of experiments results show that this method outperforms other baseline methods in UIE field.



1. This method has excellent performence over other baselines through the experiments.

2. This unsupervised framework is well-arranged, which is the first to combine traditional approaches like the underwater dark channel prior (DCP) with prompt learning in UIE task.

   

3. The DCP method, FLIP model and U-net seems to be aggregated intentionally. The motivation of how the framework is built should be declared more.

4. It seems the framework without U-net still gets a descent performence using traditional method DCP. The significance and prospects of prompt learning in UIE tasks should be emphasized besides the role of a U-net guider.

5. Fig.1 only demonstrates training stage 2, which freezes the weights in FLIP, which may cause misunderstanding because these weights are trained during stage 1.

6. Why is MSE loss computed between original image and enhanced image while SSIM loss takes into account the restored image?

7. The symbols such as Io, Ie, Pl, Ph are somehow hard to memorize. It's better to unify the subscripts.

8. The weight alpha for each encoder layer used in computing MSE loss is not provided.

9. In all, it's an excellent work on combining DCP method with prompt learning in UIE task.



为啥他用flip而不是简单的图像分类器？

两个结合能更好

N等于多少？





xiugai1tu1





首先提出 同故宫这个模块 提出什么模块和问题要关联6

是否能并行化，现在二阶段满

即插即用，人类医生专家

1.学长我取region是保证优先临近聚类，还是优先聚类后patch差不多数量呢
2.如果说canvas检测出来一个框跨越了两个patch，那这个框怎么算
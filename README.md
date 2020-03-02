# 实验设备实体词识别/序列标注模型
>模型架构采用Seq2Seq,创新点有二：
* 先使用DT Cell 双向抽取、经过MaxPool/EveragePool得到全局表征
* 使用改进的GRU Cell  ==>  DT Cell(Deep Transition cell;来自腾讯2019) 处理时序问题更强
##架构图
![](G.jpg)

参考资料：
- https://blog.csdn.net/qq_35687547/article/details/102172775
- https://github.com/pbloem/former/blob/master/experiments/classify.py
- https://github.com/pbloem/former/blob/master/former/modules.py

# 进度

## 21/10/27
完成pipeline搭建, 实现vanilla_attention。

## 21/10/28
块化文本分类函数, 添加argparser, right product。

## 21/10/29
实现linear attention, 完成cross product的正确性验证。

## 21/11/3
实现rfa。

## 21/11/10
实现performer。

# to do

- 添加测速的代码;
- q, k为负数时, 右乘和左乘的误差特别大;
- 给不同的attention适配不同的初始化。


# requirement

torchtext
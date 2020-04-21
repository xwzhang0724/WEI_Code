
# --------softmax回归------多类分类模型---------

# softmax函数是将每个类别的打分转换为合理的概率值
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 创建x，x是一个占位符（placeholder）,代表待识别的图片
x = tf.placeholder(tf.float32, [None, 784])

# 这里报错AttributeError: module 'tensorflow' has no attribute 'placeholder'
# 解决办法：使用 import tensorflow.compat.v1 as tf
#               tf.disable_v2_behavior()
# 来替换 import tensorflow as tf

# 参数变量W 784*10 b 10*1
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# y 表示模型的输出
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_ 是实际的图像标签，以占位符表示
y_ = tf.placeholder(tf.float32, [None, 10])

# -------------------------------------
# 占位符和变量的区别
# 实际上都是tensor（张量），占位符x = tf.placeholder(tf.float32, [None, 784])，无需赋予初始值
# 其中形状是[None,placeholder 784]，None表示这一维的大小可以是任意的，也可以说是传递的训练样本的个数
# 创建变量时，需要赋予初始值，变量W = tf.Variable(tf.zeros([784, 10]))，是个784*10的零矩阵
# --------------------------------------

# 根据y和y_来定义交叉熵损失
# 使用交叉熵损失来衡量y和y_的相似性
# reduce_mean 计算平均值
# reduce_sum 求和
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# 使用梯度下降法对模型的参数进行优化
# tensorflow中自带训练阶段（train）使用的优化方法梯度下降（GradientDescentOptimizer）
# 0.01为学习率
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 创建一个Session（会话），只有在Session中才能运行优化步骤train_step
sess = tf.InteractiveSession()

# 运行之前必须要初始化所有变量，分配内存
tf.global_variables_initializer().run()

# ----------------------------------
# Session 会话
# Tensor是TensorFlow进行计算的“结点”，而会话是对这些结点进行计算的上下文
# 变量是在计算过程中可以改变值的Tensor，而这些变量的值则保存在会话中
# 对变量进行操作要对这些变量进行初始化，实际上是在会话中保存变量的初始值
# 初始化语句 tf.global_variables_initializer().run()
# ----------------------------------

# 有了Session，下面对W,b进行优化，1000步梯度下降
for i in range(1000):
    # 在mnist.train中取100个训练数据
    # batch_xs (100, 784)的图像数据 batch_ys (100, 10)的实际标签
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 在Session中运行train_step,运行时要传入占位符的值
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 一些定义
# 正确的预测结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 预测准确率的计算
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 传入训练完的sess中
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# -----------------------------
# tf.argmax(y, 1), tf.argmax(y_, 1) 取出数组中最大值的下标
# 通过tf.equal来比较是否相等 True or False
# 使用tf.cast(correct_prediction, tf.float32)将值转为float类型 True--1 False--0
# 使用tf.reduce_mean来计算模型预测的准确率的平均值
# --------------------------------

# ----------------------------------------
# 在softmax回归中
#    梯度下降(0.92)优于Adam算法(0.85)
# ----------------------------------


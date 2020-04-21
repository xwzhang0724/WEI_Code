
# -------------两层卷积网络实现手写数字识别-------------------

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data

# 读入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# x为训练图像的占位符 y_为训练标签的占位符
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 将单张图片从784维向量还原为28*28的矩阵图片
x_image = tf.reshape(x, [-1, 28, 28, 1])

def weight_variable(shape):
    # 初始化W（卷积核）形状（传入的shape）和参数（初始为0.1）
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积函数
# strides 步长1步
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化函数
# ksize 池化窗口大小 2*2  strides 步长2步
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第一层卷积层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 进行卷积计算， relu作为激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第一层全连接层
# 输出为1024维的向量
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 使用Dropout， keep_prob是一个占位符，训练时为0.5， 测试时为1
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第二层全连接层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 定义交叉熵损失
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# 定义梯优化算法
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
# 定义测试准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建Session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 训练20000步
for i in range(20000):
    batch = mnist.train.next_batch(50)
    # 每100步报告一次在验证集上的准确率
    if i % 100 == 0:
        # batch[0]是训练图像 batch[1]是训练图像的标签
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        # 与sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})等价
        # %g 为浮点数字
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    # 也可以写成sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 训练结束后 在测试集上计算准确率
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


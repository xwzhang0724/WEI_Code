
# 加载mnist数据集
# 从tensorflow.examples.tutorials.mnist中引入模块
from tensorflow.examples.tutorials.mnist import input_data
# 从MNIST_data/中读取数据集，若没有则进行自动下载
# 由于网络原因下载出错，自行下载MNIST数据集，手动放入MNIST_data文件夹下
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
# 查看数据大小 28*28
# mnist.train 是训练图片数据 mnist.validation 是验证图片数据 mnist.test 是测试图片数据
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)




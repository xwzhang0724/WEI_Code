# --------将mnist数据集保存为图片--------


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
# 打印第0张训练图片对应的向量表示
print(mnist.train.images[0, :])

# 打印图片
# import scipy.misc
import os
from PIL import Image

# 将原始图片保存在raw文件夹下
save_dir = 'MNIST_data/raw/'
# 判断raw文件夹是否存在
if os.path.exists(save_dir) is False:
    # 创建文件夹
    os.makedirs(save_dir)

# 保存前20张图片
for i in range(20):
    image_array = mnist.train.images[i, :]
    # TensorFlow中的图像时一个784维的向量，这里将其还原为28*28的矩阵
    image_array = image_array.reshape(28, 28)
    # 保存文件格式为： mnist_train_0.jpg
    filename = save_dir + 'mnist_train_%d.jpg' % i
#     保存为图片
#     先用scipy.misc.toimage转换为图像，在调用save直接保存
#     scipy.misc.toimage(image_array, cmin = 0.0, cmax = 1.0).save(filename)

# 这里报错AttributeError: module 'scipy.misc' has no attribute 'toimage'
# scipy.misc.toimage这个函数将在scipy 1.2.0版本之后删除
# 使用Pillow的Image.formarray替代
    Image.fromarray((image_array*255).astype('uint8'), mode='L').save(filename)
#     Image.fromarray对float支持不好，这里可以把（0-1）的float数据乘上255，
#     之后再转成unit8就可以了，最后将mode设置成‘L’就是想要的图片
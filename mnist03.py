
# -----------图像标签的独热（one-hot）表示------------

# 所谓独热表示，就是“一位有效编码” 。
# 我们用N维的向量来表示N个类别，每个类别占据独立的一位，
# 任何时候独热表示中只有一位是1，其它都为0。

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# 打印训练图片的标签
print(mnist.train.labels[0, :])

# 显示前20张训练图片的标签
for i in range(20):
    one_hot_label = mnist.train.labels[i, :]
    # 由于是独热显示，只有一个1
    # 使用np.argmax,可以直接获得原始的label
    # 返回最大值的位置
    label = np.argmax(one_hot_label)
    print('mnist_train_%d.jpg label: %d' % (i, label))

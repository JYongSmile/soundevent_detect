# 有几个卷积核就有几个输出的feature map,叫特征图，不同眼睛不同视野
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# 模型评估
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# 第一步：
def weight_variable(shape):  ##定义一个权重初始化函数，大数据循环，避免每次都要初始化，目的产生随机变量w,这里是指卷积核
    initial = tf.truncated_normal(shape,
                                  stddev=0.1)  # 变量W是使用tf.truncanted_normal输出服从截尾正态分布的随机值，z这么做的原因：若初始值为0，则a1=a2=a2......，大量的w值只能产生一个输入了,无意义。
    return tf.Variable(initial)


def bias_variable(shape):  ##def一个偏置初始化函数，目的产生常量b
    initial = tf.constant(0.1, shape=shape)  # 使用tf.constant函数来产生tensor常量b，初始值0.1，属性shape
    return tf.Variable(initial)  # 输出这个变量


def conv2d(x, W):  # 定义卷积方法
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                        padding='SAME')  # 定义卷积方法，tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)，输入图片、卷积核、步长、我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding
    'SAME')  # pooling池化，跟卷积相似但是不一样，pooling只是一个窗口在非线性激活后的featrue map上运动，每次运动的格子没有重复或覆盖，提取被窗口覆盖的数值的最大值，得到一张池化后的feature map，数据量变小。因不想在batch和channels上做池化，则将其值设为1。

    # 1.cnn:先重新定义一下输入和输出
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)  # 设置概率keep_prob，为drop层定义一个占位符，用来防止过拟合，tf实现dropout其实,就一个函数,让一个神经元以某一固定的概率失活
    x_image = tf.reshape(xs, [-1, 28, 28,
                              1])  # 把输入x(二维张量,shape为[batch, 784])变成4d的x_image，x_image的shape应该是[batch,28,28,1],batch就是跟None一样，输入维度多少不知道，一次输入的是多个数据图片，而不是一个图片，这里用-1表示，输进去多少图片，在这个方法下会自动检测batch有多少

    ## conv1 layer ##
    # 定义第一层卷积层
    W_conv1 = weights_variable([5, 5, 1,
                                32])  # 注意，之前的只是初始化函数，现在是定义这个层，卷积层：用5*5的卷积核去卷积，输入的是灰度故size=1，用32个卷积核卷积，输出的是32个feature map（√）  [filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
    b_conv1 = bias_variable([32])  # 定义bias，注意：有几个卷积核W，就有几个b，如y = Wx + b，这里都是32
    h_conv1 = tf.nn.relu(conv2d(x_image,
                                W_con1) + b_conv1)  # 用之前定义好的convenience2d卷积，把变量输进去，在tf.nn.relu下输出值，并返回给h_conv1代表第一层卷积的结果。注意conv2d(x_image,W_con1) + b_conv，这部分就是卷积的代码，生成一张feature map的粗糙形式，然后再非线性激活一下，然后再pooling，tf.nn.relu就是非线性激活函数，所以，卷积结果要放进（）去激活一下.
    # 输出形式28*28*32，之前输入的xs是784的，经过x_imge后变成28*28，在经过卷积核对应乘积后，按理说变成 Q*Q形式，Q<28，但是我们用的padding是SAME，即窗口和原始的x_imge一样的大，即Q*Q周围都补上0，窗口是一样大。输出也是32个数字，在一张map中
    h_pool1 = max_pool_2x2(h_conv1)  # 用maxpool函数将激活后的卷积feature map 池化一下，保存到h_pool1中。
    # 输出形式14*14*32，由于卷积时的步长是【1，1，1，1】，而pooling的是1，2，2，1 所以，变成每隔两个步长提取一个最大数字，所以，得到的会是缩减维度各缩减一半的feature map.故是28/2，28/2，而输出不变仍然是32个卷积核作用的结果，故有32个输出
    # 采用最大池化，也就是取窗口中的最大值作为结果
    # x 是一个4维张量，shape为[batch,height,width,channels]
    # ksize表示pool窗口大小为2x2,也就是高2，宽2
    # strides，表示在height和width维度上的步长都为2


    ## conv2 layer ##
    ##定义第二次卷积层
    W_conv2 = weights_variable([5, 5, 32, 64])  # 将第一层的输出部分作为第二层的输入部分，第二层用64个卷积核，有64个输出
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 输出格式：14，14，64 之前的pooling输出是14*14*32
    h_pool2 = max_pool_2x2(h_conv2）  # 输出格式经过2*2pooling后再缩小一半：7，7，64
    ## func1 layer ##
    # 定义第一层全连接层
    # 这层是拥有1024个神经元的全连接层
    # W的第1维size为7*7*64，7*7是h_pool2输出的size，64是第2层输出神经元个数
    W_fc1 = weights_variable([7 * 7 * 64, 1024])  # 全连接层就要铺展开成一维形式，也就是原来的卷积层的三维要变成一维形式，故，定义W时如沿用卷积层的形式，稍作改变
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1,
                                        7 * 7 * 64])  # 把第二卷积层的输出结果作为全连接层的输入层，而，全链接是一维形式的，故用tf.reshape的方法去将三维的转化成移位的，即[7,7,64]->>[7*7*64]
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat,
                                 W_fc1) + b_fc1)  # 将第二层yiwei化后，加权、偏置，算是一项处理，matmul实现最基本的矩阵相乘，，W_fc1单行乘以h_pool单列等于1*1矩阵，就是一个数吧，不同于tf.nn.conv2d的遍历相乘，自动认为是前行向量*后列向量
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 定义为防止过拟合 ————keep_prob：任何一个给定单元的留存率（没有被丢弃的单元）设置概率keep_prob

    # 第二个全连接层
    ## func2 layer ##
    W_fc2 = weights_variable([1024, 10])  ##上一层的全链接输出作为本层的输入是1024，因为是数字识别，故有10位数字，10位输出
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 加权加偏置后，用softmax回归算法分配概率，接下来跟mnist基本一样了。

    # 用交叉熵计算真实值和预测值之间的差值
    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    # 用
    train_step = tf.train.AdamOptimizer(1e-4).minimize(
        cross_entropy)  # 使用AdamOptimizer（是一个寻找全局最优点的优化算法，引入了二次方梯度校正）引导导向loss不断减小的方向，学习速率：就科学计数法，即1乘以10的-4次方。注意这里的1不能省略，因为可能造成歧义

    sess = tf.Session()  # 初始化session
    # important step
    sess.run(tf.initialize_all_variables())  # 激活所有变量

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
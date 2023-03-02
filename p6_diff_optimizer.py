# 引入需要的各种包
from sklearn import datasets
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import time  # 1 相比classing_iris.py 引入时间模块以计算处理速度

# 加载数据库
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 打乱是为了避免按照数据库学习
np.random.seed(116)  # 使用同样的随机种子，使得打乱的方式一样
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train = x_data[:-30]  # 30以前为训练集，后30为测试集
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)  # 转化为tensorflow能够处理的形式
x_test = tf.cast(x_test, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
# 将x与y（在这里 是特征与标签）以32为一组打包
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 生成可训练参数，注意tf的所有都是T的
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))  # 输入层有4个变量，输出层有3个变量
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))  # 分别是y=wx+b中的w和b

# 定义数值方法部分
lr = 0.1
train_loss_results_adagard = []  # 损失函数表
test_acc = []  # 准确率表
epoch = 500  # 循环500轮
loss_all = 0  # loss 函数batch和

v_w, v_b = 0, 0  # Adagard

for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1  # y
            y = tf.nn.softmax(y)  # 概率
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))  # 均方差
            loss_all += loss.numpy()
        grads = tape.gradient(loss, [w1, b1])

        v_w += tf.square(grads[0]) # Adagrad
        v_b += tf.square(grads[1])
        w1.assign_sub(lr * grads[0] / tf.sqrt(v_w))
        b1.assign_sub(lr * grads[1] / tf.sqrt(v_b))

    print("Epoch {}, loss: {}".format(epoch, loss_all / 4))
    train_loss_results_adagard.append(loss_all / 4)
    loss_all = 0
    print("-----------------------------------")


# 加载数据库
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 打乱是为了避免按照数据库学习
np.random.seed(116)  # 使用同样的随机种子，使得打乱的方式一样
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train = x_data[:-30]  # 30以前为训练集，后30为测试集
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)  # 转化为tensorflow能够处理的形式
x_test = tf.cast(x_test, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
# 将x与y（在这里 是特征与标签）以32为一组打包
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 生成可训练参数，注意tf的所有都是T的
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))  # 输入层有4个变量，输出层有3个变量
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))  # 分别是y=wx+b中的w和b

# 定义数值方法部分
lr = 0.1
train_loss_results_sgd = []  # 损失函数表
test_acc = []  # 准确率表
epoch = 500  # 循环500轮
loss_all = 0  # loss 函数batch和

now_time = time.time()  # 2 记录训练的起始时间
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1  # y
            y = tf.nn.softmax(y)  # 概率
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))  # 均方差
            loss_all += loss.numpy()
        grads = tape.gradient(loss, [w1, b1])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
    print("Epoch {}, loss: {}".format(epoch, loss_all / 4))
    train_loss_results_sgd.append(loss_all / 4)
    loss_all = 0
    print("-----------------------------------")


# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样（为方便教学，以保每位同学结果一致）
np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 生成神经网络的参数，4个输入特征故，输入层为4个输入节点；因为3分类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
# 使用seed使每次生成的随机数相同（方便教学，使大家结果都一致，在现实使用时不写seed）
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1  # 学习率为0.1
train_loss_results_adam = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
epoch = 500  # 循环500轮
loss_all = 0  # 每轮分4个step，loss_all记录四个step生成的4个loss的和

##########################################################################
m_w, m_b = 0, 0
v_w, v_b = 0, 0
beta1, beta2 = 0.9, 0.999
delta_w, delta_b = 0, 0
global_step = 0
##########################################################################

# 训练部分
now_time = time.time()  ##2##
for epoch in range(epoch):  # 数据集级别的循环，每个epoch循环一次数据集
    for step, (x_train, y_train) in enumerate(train_db):  # batch级别的循环 ，每个step循环一个batch
 ##########################################################################
        global_step += 1
 ##########################################################################
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算loss和accuracy
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])

##########################################################################
 # adam
        m_w = beta1 * m_w + (1 - beta1) * grads[0]
        m_b = beta1 * m_b + (1 - beta1) * grads[1]
        v_w = beta2 * v_w + (1 - beta2) * tf.square(grads[0])
        v_b = beta2 * v_b + (1 - beta2) * tf.square(grads[1])

        m_w_correction = m_w / (1 - tf.pow(beta1, int(global_step)))
        m_b_correction = m_b / (1 - tf.pow(beta1, int(global_step)))
        v_w_correction = v_w / (1 - tf.pow(beta2, int(global_step)))
        v_b_correction = v_b / (1 - tf.pow(beta2, int(global_step)))

        w1.assign_sub(lr * m_w_correction / tf.sqrt(v_w_correction))
        b1.assign_sub(lr * m_b_correction / tf.sqrt(v_b_correction))
##########################################################################

    # 每个epoch，打印loss信息
    print("Epoch {}, loss: {}".format(epoch, loss_all / 4))
    train_loss_results_adam.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备
    print("--------------------------")






plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results_adagard, label="$Loss-adagard$")
plt.plot(train_loss_results_adam, label="$Loss-adam$")
plt.plot(train_loss_results_sgd, label="$Loss-sgd$")
plt.legend()
plt.show()


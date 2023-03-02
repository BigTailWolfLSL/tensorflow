# 引入需要的各种包
from sklearn import datasets
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

#加载数据库
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 打乱是为了避免按照数据库学习
np.random.seed(116) # 使用同样的随机种子，使得打乱的方式一样
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train = x_data[:-30] #后30以前为训练集，后30为测试集
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型，便于后面矩阵相乘运算
x_train = tf.cast(x_train, tf.float32) #转化为tensorflow能够处理的形式
x_test = tf.cast(x_test, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
# 将x与y（在这里 是特征与标签）以32为一组打包
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

#生成可训练参数，注意tf的所有都是T的
w1 = tf.Variable(tf.random.truncated_normal([4,3], stddev=0.1, seed=1)) # 输入层有4个变量，输出层有3个变量
b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.1, seed=1)) # 分别是y=wx+b中的w和b

# 定义数值方法部分
lr = 0.1
train_loss_results= [] # 损失函数表
test_acc = [] # 准确率表
epoch = 500 # 循环500轮
loss_all = 0 # loss 函数batch和

for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1 #y
            y = tf.nn.softmax(y) #概率
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_-y)) # 均方差
            loss_all += loss.numpy()
        grads = tape.gradient(loss, [w1,b1])
        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr*grads[1])
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all/4)
    loss_all = 0
    print("-----------------------------------")


##decay_lr 部分
#加载数据库
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 打乱是为了避免按照数据库学习
np.random.seed(116) # 使用同样的随机种子，使得打乱的方式一样
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train = x_data[:-30] #后30以前为训练集，后30为测试集
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型，便于后面矩阵相乘运算
x_train = tf.cast(x_train, tf.float32) #转化为tensorflow能够处理的形式
x_test = tf.cast(x_test, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
# 将x与y（在这里 是特征与标签）以32为一组打包
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

#生成可训练参数，注意tf的所有都是T的
w1 = tf.Variable(tf.random.truncated_normal([4,3], stddev=0.1, seed=1)) # 输入层有4个变量，输出层有3个变量
b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.1, seed=1)) # 分别是y=wx+b中的w和b

# 定义数值方法部分
train_loss_results2= [] # 损失函数表
test_acc = [] # 准确率表
epoch = 500 # 循环500轮
loss_all = 0 # loss 函数batch和

LR_BASE = 0.5  # decay_lr 最初学习率
LR_DECAY = 0.99  # 学习率衰减率
LR_STEP = 1  # 喂入多少轮BATCH_SIZE后，更新一次学习率

for epoch in range(epoch):
    lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1 #y
            y = tf.nn.softmax(y) #概率
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_-y)) # 均方差
            loss_all += loss.numpy()
        grads = tape.gradient(loss, [w1,b1])
        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr*grads[1])
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results2.append(loss_all/4)
    loss_all = 0
    print("-----------------------------------")





plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label="$Loss-lr=0.1$")
plt.plot(train_loss_results2, label="$Loss-decaylr=0.5*0.99^n$")
plt.legend()
plt.show()


#整个程序为
#准备数据：1，数据集读入；2，数据集乱序；3，生成训练集和测试集；4，配对输入和标签，每次读入一笑包batch；
#搭建网络：定义神经网络中所有可训练的参数
#参数的优化：嵌套循环迭代，with结构更新参数，显示当前loss
#测试效果：计算当前前向传播后的准确率，显示当前acc
#acc或者loss变化曲线图，（acc是测试集的准确率）
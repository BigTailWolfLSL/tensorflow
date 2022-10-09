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

x_train = x_data[:-30] #30以前为训练集，后30为测试集
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

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

    total_correct, total_number =0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1  # 计算测试y 原因在于tf是行向量，考虑到转置（wx）T=xTwT
        y = tf.nn.softmax(y)  # 转化为概率
        pred = tf.argmax(y, axis=1)  # 返回最大概率的类别
        pred = tf.cast(pred, dtype=y_test.dtype)  # 把数据类型转化为结果类型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)  # 将包里面的正确预测数量加和
        total_correct += int(correct)  # 汇总不同包的总数
        total_number += x_test.shape[0]
    acc = total_correct / total_number  # 正确率
    test_acc.append(acc)
    print("test_acc:", acc)
    print("-----------------------------------")

plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label="$Loss$")
plt.legend()
plt.show()

plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc,label="$Accuracy")
plt.legend()
plt.show()
import tensorflow as tf  # 全选Ctrl+Alt+L自动排版
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TKAgg')  # 为了让图显示出来
import numpy as np
import pandas as pd

df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_train = np.vstack(x_data).reshape(-1, 2)
y_train = np.vstack(y_data).reshape(-1, 1)

Y_c = [['red' if y else 'blue'] for y in y_train]

x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# 一个隐藏层11个神经元，共三层
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[11]))

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.005
epoch = 800

for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1 = tf.matmul(x_train, w1) + b1  # 隐藏层
            h1 = tf.nn.relu(h1)  # 非线性因素，relu函数
            y = tf.matmul(h1, w2) + b2  # 输出层不加激活函数
            loss = tf.reduce_mean(tf.square(y_train - y))  # 均方差损失函数
        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)

        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    if epoch % 20 == 0:
        print('epoch:', epoch, 'loss:', float(loss))

print("*******predict*******")
xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)

probs = []
for x_test in grid:
    h1 = tf.matmul([x_test], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2
    probs.append(y)

x1 = x_data[:, 0]
x2 = x_data[:, 1]
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1, x2, color=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])  # 这个函数用来画等高线的，前两个参数网格xy坐标，第三个参数是高度，第四个是要画等高线的高度，因为这里只有0和1可以这样用
plt.show()

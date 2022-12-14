import tensorflow as tf
import numpy as np

SEED = 23455

rdm = np.random.RandomState(seed=SEED)
x = rdm.rand(32, 2) #随机生成数据集，伪造一个y=x1+x2的关系
y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in x]
x = tf.cast(x, dtype=tf.float32)

w1 = tf.Variable (tf.random.normal([2,1],stddev=1,seed=1))

epoch = 15000
lr = 0.002

Cost=99
Profit=1
for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y=tf.matmul(x,w1)
        zidingyi_loss = tf.reduce_sum(tf.where(tf.greater(y,y_), (y-y_)*Cost, (y_-y)*Profit))

    grads = tape.gradient(zidingyi_loss, w1)
    w1.assign_sub(lr * grads)

    if epoch % 500 == 0:
        print("After %d training steps,w1 is " % (epoch))
        print(w1.numpy(), "\n")
print("Final w1 is: ", w1.numpy())
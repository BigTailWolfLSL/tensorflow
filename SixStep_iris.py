import tensorflow as tf  # 1 import
from sklearn import datasets
import numpy as np

x_train = datasets.load_iris().data  # 2 train test
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

model = tf.keras.models.Sequential([  # 3 models.Sequential
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])# 搭建神经网络：输出神经元个数，选用的激活函数，选用的正则化方法

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),  # 4 model.compile
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
# 配置训练方法：选择SGD优化器学习率=0.1，SCC为损失函数（因为神经网络使用了softmax，输出结果是概率，选择False），结果是分类数值但输出是概率使用数值-》概率的SCA

model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)  # 5 model.fit
#执行训练过程：训练集变量，训练集结果，训练时一次喂入包的大小，迭代次数，VS从训练集中选择20%的数据作为测试集，VF每迭代20次在测试集中验证一次准确率

model.summary()  # 6 model.summary

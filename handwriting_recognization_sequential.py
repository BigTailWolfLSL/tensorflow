import tensorflow as tf
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist  # 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[0], cmap='gray')  # 画出第一个，灰度图
plt.show()

# 打印出输入输出的样子
print("x_train[0]:\n", x_train[0])
print("y_train[0]:\n", y_train[0])

# 打印shape
print("x_train.shape:\n", x_train.shape)
print("y_train.shape:\n", y_train.shape)
print("x_test.shape:\n", x_test.shape)
print("y_test.shape:\n", y_test.shape)

x_train = x_train / 255.0
x_test = x_test / 255.0  # 数值变小更合适，更好地学习

model = tf.keras.models.Sequential([  # 3层表示三层操作
    tf.keras.layers.Flatten(),  # 第一层拉平
    tf.keras.layers.Dense(128, activation='relu'),  # 第二层是第一层神经网络，128个结果，relu激活函数
    tf.keras.layers.Dense(10, activation='softmax')  # 第三层是第二层神经网络，10个结果，softmax输出层
])  # 搭建神经网络

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # softmax了
              metrics=['sparse_categorical_accuracy']
              )  # 配置训练方法，优化器，损失函数，评测指标

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
# 执行训练过程

model.summary()

#用新定义类的方法实现
'''import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y


model = MnistModel()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
model.summary()'''

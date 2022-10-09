from PIL import Image
import numpy as np
import tensorflow as tf

model_save_path = './checkpoint/mnist.ckpt'

model = tf.keras.models.Sequential([  # 复现模型，搭建对应的神经网络
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights(model_save_path)  # 加载参数

image_path = input("the path of test picture:")
img = Image.open(image_path)
img = img.resize((28, 28), Image.ANTIALIAS)  # 变成对应大小的灰度图片
img_arr = np.array(img.convert('L'))  # 转化为数值矩阵
img_arr = 255 - img_arr  # 转化为黑底白字，符合训练的情况：预处理

'''
for i in range(28):
    for j in range(28):
        if img_arr[i][j] < 200:
            img_arr[i][j]=255
        else:
            img_arr[i][j]=0
'''  # 备选预处理方法，转化为高对比度的图片

img_arr = img_arr / 255.0
x_predict = img_arr[tf.newaxis, ...]  # 因为训练时都是按照batch送入，多一个维度
result = model.predict(x_predict)
print(result.shape)
pred = tf.argmax(result, axis=1)  # 选出最大值的位置，axis表示比较的方向,axis=0列最大值下标，axis=1行最大值下标
print(pred)
tf.print(pred)

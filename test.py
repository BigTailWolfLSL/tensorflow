import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tensorflow_version = tf.__version__
gpu_available = tf.test.is_gpu_available()

print("tensorflow version:", tensorflow_version, "\tGPU available:", gpu_available)

a = tf.constant([1.0,2.0], name="a")
b = tf.constant([1.0,2.0], name="b")
result = tf.add(a,b,name="add")
print(result)

import tensorflow as tf

w=tf.Variable(tf.constant(5, dtype=tf.float32))
lr=0.4
epoch=40

for epoch in range(epoch): # for 1:epoch
    with tf.GradientTape() as tape: # pack up
        loss=tf.square(w+1) # loss function
    grads=tape.gradient(loss,w) # partial derivative

    w.assign_sub(lr * grads) # w-=lr*grads
    print("After %s epoch, w is %f, loss is %f" % (epoch, w.numpy(), loss))

# lr 0.01 收敛慢 0.99 不收敛
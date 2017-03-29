# observe the response of local CNN model for other class data
# observe response of local model for other class data
import os
import tarfile
import gzip
import tensorflow as tf
import numpy
from tensorflow.contrib.session_bundle import exporter

tf.logging.set_verbosity(tf.logging.DEBUG)
sess = tf.InteractiveSession()
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images = mnist.train._images
train_labels = mnist.train._labels
print(train_images.shape)



test_images=mnist.test._images
test_labels=mnist.test._labels
print(test_images.shape)


trainrawinput = tf.constant(numpy.array(test_images), name='trainrawinput')
trainrawlabels = tf.constant(numpy.array(test_labels), name='trainrawlabels')


# load 1st model
model_dir='CNNmnistModel/mnist-model'
new_saver = tf.train.import_meta_graph(model_dir+str(5)+'.meta')
new_saver.restore(sess, model_dir+str(5))
x1=sess.graph.get_tensor_by_name('x:0')
y_conv1=sess.graph.get_tensor_by_name('y_conv:0')
keep_prob1=sess.graph.get_tensor_by_name('keep_prob:0')

predictions_localmodel1 = sess.run(y_conv1,feed_dict={x1:test_images, keep_prob1:1.0})
#print(predictions_localmodel1)


# labels without one-hot encoding
num_examples=trainrawlabels.get_shape()[0]
print(num_examples)
purelabels=tf.cast(tf.argmax(trainrawlabels,1),tf.int32)
testlabelpure=tf.reshape(tf.argmax(trainrawlabels,1),[10000,1])

# predictions without one-hot encoding
purepred_localmodel1=sess.run(tf.cast(tf.argmax(predictions_localmodel1,1),tf.int32))
compare = tf.concat([predictions_localmodel1, tf.cast(testlabelpure,tf.float32)],1)

purelabels2=sess.run(purelabels)
print(purepred_localmodel1[0])
class0count=0
class1count=0
for i in range(0,10000):
    if(purelabels2[i]==0):
        if(purepred_localmodel1[i]==0):
            class0count=class0count+1
        else:
            class1count=class1count+1

print(class0count)
print(class1count)

class0count=0
class1count=0
for i in range(0,10000):
    if(purelabels2[i]==1):
        if(purepred_localmodel1[i]==0):
            class0count=class0count+1
        else:
            class1count=class1count+1

print(class0count)
print(class1count)

class0count=0
class1count=0
for i in range(0,10000):
    if(purelabels2[i]==2):
        if(purepred_localmodel1[i]==0):
            class0count=class0count+1
        else:
            class1count=class1count+1

print(class0count)
print(class1count)

class0count=0
class1count=0
for i in range(0,10000):
    if(purelabels2[i]==3):
        if(purepred_localmodel1[i]==0):
            class0count=class0count+1
        else:
            class1count=class1count+1

print(class0count)
print(class1count)

class0count=0
class1count=0
for i in range(0,10000):
    if(purelabels2[i]==4):
        if(purepred_localmodel1[i]==0):
            class0count=class0count+1
        else:
            class1count=class1count+1

print(class0count)
print(class1count)

class0count=0
class1count=0
for i in range(0,10000):
    if(purelabels2[i]==5):
        if(purepred_localmodel1[i]==0):
            class0count=class0count+1
        else:
            class1count=class1count+1

print(class0count)
print(class1count)

class0count=0
class1count=0
for i in range(0,10000):
    if(purelabels2[i]==6):
        if(purepred_localmodel1[i]==0):
            class0count=class0count+1
        else:
            class1count=class1count+1

print(class0count)
print(class1count)

class0count=0
class1count=0
for i in range(0,10000):
    if(purelabels2[i]==7):
        if(purepred_localmodel1[i]==0):
            class0count=class0count+1
        else:
            class1count=class1count+1

print(class0count)
print(class1count)

class0count=0
class1count=0
for i in range(0,10000):
    if(purelabels2[i]==8):
        if(purepred_localmodel1[i]==0):
            class0count=class0count+1
        else:
            class1count=class1count+1

print(class0count)
print(class1count)

class0count=0
class1count=0
for i in range(0,10000):
    if(purelabels2[i]==9):
        if(purepred_localmodel1[i]==0):
            class0count=class0count+1
        else:
            class1count=class1count+1

print(class0count)
print(class1count)
# print(compare.eval())
# tf.Print(trainrawinput,[compare.eval()])
numpy.set_printoptions(precision=2)
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(threshold=numpy.nan)
#print(numpy.array(sess.run(compare)))

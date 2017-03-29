# level1 output(without raw input data) to level2
# tuning learning rate
import os
import tarfile
import gzip
import tensorflow as tf
import numpy
import time
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
#import localmodel_inf

tf.logging.set_verbosity(tf.logging.DEBUG)
config = tf.ConfigProto(allow_soft_placement=True)

sess = tf.InteractiveSession(config=config)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#print(mnist.train.images)
train_images=mnist.train._images
train_labels=mnist.train._labels
#print(train_images.shape)
# print(train_images)
# print("Train Labels:")
# print(train_labels)


test_images=mnist.test._images
test_labels=mnist.test._labels

# print(test_images)
# print("Test Labels:")
# print(test_labels)

# Function to generate batch data in numpy array format
indexinepoch = 0
epochs_completed = 0
def next_batch(images, labels, batch_size):
    """Return the next `batch_size` examples from this data set."""
    numofexp = 55000
    global indexinepoch
    global epochs_completed
    start = indexinepoch
    indexinepoch += batch_size
    if indexinepoch > numofexp:
        # Finished epoch
        epochs_completed += 1
        # Shuffle the data
        perm = numpy.arange(numofexp)
        numpy.random.shuffle(perm)
        images = images[perm]
        labels = labels[perm]
        # Start next epoch
        start = 0
        indexinepoch = batch_size
        assert batch_size <= numofexp
    end = indexinepoch
    return images[start:end], labels[start:end]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def local_model(class1, class2, model_num):
    train_images_class1 = []
    train_images_class2 = []
    train_labels_originals_class1=[]
    train_labels_originals_class2 = []
    trainimages_classes=[]
    trainlabels_classes = []
    j = 0
    for i in train_labels:
        if (i[class1] == 1):
            train_images_class1.append(train_images[j])
            train_labels_originals_class1.append(i)
            trainimages_classes.append(train_images[j])
            trainlabels_classes.append([1,0])
        if (i[class2] == 1):
            train_images_class2.append(train_images[j])
            train_labels_originals_class2.append(i)
            trainimages_classes.append(train_images[j])
            trainlabels_classes.append([0, 1])
        j = j + 1

    #print(train_labels_originals_class1)
    class1input = tf.Variable(numpy.array(train_images_class1), name='class1input', dtype="float32")
    class2input = tf.Variable(numpy.array(train_images_class2), name='class2input', dtype="float32")
    #print(class1input)
    class1label_1 = tf.Variable(tf.ones([class1input.get_shape()[0], 1]), name='class1label')
    class1label_2 = tf.Variable(tf.zeros([class1input.get_shape()[0], 1]), name='class1label')
    sess.run(tf.variables_initializer([class1label_1, class1label_2]))
    class1label = tf.concat( [class1label_1, class1label_2],1)

    class2label_1 = tf.Variable(tf.zeros([class2input.get_shape()[0], 1]), name='class2label')
    class2label_2 = tf.Variable(tf.ones([class2input.get_shape()[0], 1]), name='class2label')
    sess.run(tf.variables_initializer([class2label_1, class2label_2]))
    class2label = tf.concat( [class2label_1, class2label_2],1)

    #convolution network
    x = tf.placeholder(tf.float32, shape=[None, 784],name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    trainimages=tf.concat([class1input,class2input],0)
    trainlabels=tf.concat([class1label,class2label],0)
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # readout layer
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name='y_conv')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    start = int(round(time.time() * 1000))
    for i in range(100):
        #batch = mnist.train.next_batch(50)
        #sess.run(train_step)
        if i % 100 == 0:
            print(i)
            #train_accuracy = accuracy.eval(feed_dict={x:trainimages_classes, y_: numpy.array(trainlabels), keep_prob: 1.0})
            #print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: trainimages_classes, y_: trainlabels_classes, keep_prob: 0.5})
    #print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    end = int(round(time.time() * 1000))
    print("Time for building convnet: ")
    print(end - start)

    # Evaluate individual model
    # test images
    test_images_class1 = []
    test_images_class2 = []
    testimages_classes = []
    testlabels_classes = []

    j = 0
    for i in test_labels:
        if (i[class1] == 1):
            test_images_class1.append(test_images[j])
            testimages_classes.append(test_images[j])
            testlabels_classes.append([1, 0])
        if (i[class2] == 1):
            test_images_class2.append(test_images[j])
            testimages_classes.append(test_images[j])
            testlabels_classes.append([0, 1])
        j = j + 1

    class1test = tf.constant(numpy.array(test_images_class1), name='class1test', dtype="float32")
    class2test = tf.constant(numpy.array(test_images_class2), name='class2test', dtype="float32")
    #print(class1test)
    class1testlabel_1 = tf.Variable(tf.ones([class1test.get_shape()[0], 1]), name='class1testlabel')
    class1testlabel_2 = tf.Variable(tf.zeros([class1test.get_shape()[0], 1]), name='class1testlabel')
    sess.run(tf.variables_initializer([class1testlabel_1, class1testlabel_2]))
    class1testlabel = tf.concat([class1testlabel_1, class1testlabel_2], 1)

    class2testlabel_1 = tf.Variable(tf.zeros([class2test.get_shape()[0], 1]), name='class2testlabel')
    class2testlabel_2 = tf.Variable(tf.ones([class2test.get_shape()[0], 1]), name='class2testlabel')
    sess.run(tf.variables_initializer([class2testlabel_1, class2testlabel_2]))
    class2testlabel = tf.concat([class2testlabel_1, class2testlabel_2], 1)
    y_test=numpy.array(tf.concat([class1testlabel,class2testlabel], 0))
    print("test accuracy %g" % accuracy.eval(feed_dict={x: testimages_classes, y_: testlabels_classes , keep_prob: 1.0}))

    # model export path
    export_path = 'data'+'//'
    print('Exporting trained model to', export_path)

    #
    saver = tf.train.Saver(sharded=True)
    # model_exporter = exporter.Exporter(saver)
    # model_exporter.init(
    #     sess.graph.as_graph_def(),
    #     named_graph_signatures={
    #         'inputs': exporter.generic_signature({'images': x}),
    #         'outputs': exporter.generic_signature({'scores': y_conv})})
    #
    # #model_exporter.export(export_path, tf.constant(1), sess)
    saver.save(sess, export_path + 'mnist-model'+str(model_num))

    # Write out the trained graph and labels with the weights stored as constants.
    # output_graph_def = graph_util.convert_variables_to_constants(
    #     sess, sess.graph.as_graph_def(), ['y_conv'])
    # with gfile.FastGFile(export_path+'mnist_local', 'wb') as f:
    #     f.write(output_graph_def.SerializeToString())
    # with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
    #     f.write('\n'.join(image_lists.keys()) + '\n')
#
# local_model(0,1,1)
# local_model(2,3,2)
# local_model(4,5,3)
# local_model(6,7,4)
# local_model(8,9,5)



# traininputlevel1=mnist.train.images
# traininputpredictions1=localmodel_inf.get_inference(1, traininputlevel1)
# traininputpredictions2=localmodel_inf.get_inference(2, traininputlevel1)
# traininputpredictions3=localmodel_inf.get_inference(3, traininputlevel1)
# traininputpredictions4=localmodel_inf.get_inference(4, traininputlevel1)
# traininputpredictions5=localmodel_inf.get_inference(5, traininputlevel1)
#
# level2traininput=tf.concat([traininputpredictions1,traininputpredictions2,traininputpredictions3,traininputpredictions4,traininputpredictions5],1)
with tf.device('/gpu:0'):
    # load 1st model
    model_dir='CNNmnistModel/mnist-model'
    new_saver = tf.train.import_meta_graph(model_dir+str(1)+'.meta')
    new_saver.restore(sess, model_dir+str(1))
    x1=sess.graph.get_tensor_by_name('x:0')
    y_conv1=sess.graph.get_tensor_by_name('y_conv:0')
    keep_prob1=sess.graph.get_tensor_by_name('keep_prob:0')
    print(x1,y_conv1,keep_prob1)

    # load 2nd model
    new_saver1 = tf.train.import_meta_graph(model_dir+str(2)+'.meta')
    new_saver1.restore(sess, model_dir+str(2))
    x2=sess.graph.get_tensor_by_name('x_1:0')
    y_conv2=sess.graph.get_tensor_by_name('y_conv_1:0')
    keep_prob2=sess.graph.get_tensor_by_name('keep_prob_1:0')
    print(x2,y_conv2,keep_prob2)

    # load 3rd model
    new_saver2 = tf.train.import_meta_graph(model_dir+str(3)+'.meta')
    new_saver2.restore(sess, model_dir+str(3))
    x3=sess.graph.get_tensor_by_name('x_2:0')
    y_conv3=sess.graph.get_tensor_by_name('y_conv_2:0')
    keep_prob3=sess.graph.get_tensor_by_name('keep_prob_2:0')
    print(x3,y_conv3,keep_prob3)

    # load 4th model
    new_saver3 = tf.train.import_meta_graph(model_dir+str(4)+'.meta')
    new_saver3.restore(sess, model_dir+str(4))
    x4=sess.graph.get_tensor_by_name('x_3:0')
    y_conv4=sess.graph.get_tensor_by_name('y_conv_3:0')
    keep_prob4=sess.graph.get_tensor_by_name('keep_prob_3:0')
    print(x4,y_conv4,keep_prob4)

    # load 5th model
    new_saver4 = tf.train.import_meta_graph(model_dir+str(5)+'.meta')
    new_saver4.restore(sess, model_dir+str(5))
    x5=sess.graph.get_tensor_by_name('x_4:0')
    y_conv5=sess.graph.get_tensor_by_name('y_conv_4:0')
    keep_prob5=sess.graph.get_tensor_by_name('keep_prob_4:0')
    print(x5,y_conv5,keep_prob5)

    level2traininput=tf.placeholder(tf.float32,[None,10],name='level2traininput')
    print(level2traininput)

    level2W = tf.Variable(tf.truncated_normal([10, 10],stddev=0.1),name='W')
    level2b = tf.Variable(tf.random_normal([10]),name='b')
    sess.run(tf.variables_initializer([level2W, level2b]))

    level2y = tf.nn.softmax(tf.matmul(level2traininput, level2W) + level2b,name='level2y')
    level2y_ = tf.placeholder(tf.float32, [None, 10],name='level2y_')
    #level2y_ = mnist.train.images

    level2cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=level2y, labels=level2y_))
    tf.summary.scalar('cross_entropy', level2cross_entropy)
    merged=tf.summary.merge_all()
    trainwriter=tf.summary.FileWriter('data/logs',sess.graph)

    # exponential decay for learning rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step=global_step,
                                               decay_steps=10000, decay_rate=0.5, staircase=True)

    # inverse time decay for learning rate
    # global_step = tf.Variable(0, trainable=False)
    # starter_learning_rate = 0.1
    # k = 0.5
    # learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, decay_rate=k,decay_steps=100)
    level2train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(level2cross_entropy,global_step=global_step)



    #sess.graph.finalize()
    sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()))

    traininputpredictions1 = sess.run(y_conv1, feed_dict={x1: mnist.train.images, keep_prob1: 1.0})
    traininputpredictions2 = sess.run(y_conv2, feed_dict={x2: mnist.train.images, keep_prob2: 1.0})
    traininputpredictions3 = sess.run(y_conv3, feed_dict={x3: mnist.train.images, keep_prob3: 1.0})
    traininputpredictions4 = sess.run(y_conv4, feed_dict={x4: mnist.train.images, keep_prob4: 1.0})
    traininputpredictions5 = sess.run(y_conv5, feed_dict={x5: mnist.train.images, keep_prob5: 1.0})

    level2traininput_temp = tf.concat(
        [traininputpredictions1, traininputpredictions2, traininputpredictions3, traininputpredictions4,
         traininputpredictions5], 1).eval()

    mnist.train.labels


    start = int(round(time.time() * 1000))
    for i in range(400000):
        global_step=i
        if (i%1000)==0:
            print(i)
        # batch_xs, batch_ys = mnist.train.next_batch(100)
        batch_xs, batch_ys = next_batch(level2traininput_temp, mnist.train.labels, 1000)
        # #traininputlevel1 = batch_xs
        # traininputpredictions1 = sess.run(y_conv1,feed_dict={x1:batch_xs, keep_prob1:1.0})
        # traininputpredictions2 = sess.run(y_conv2,feed_dict={x2:batch_xs, keep_prob2:1.0})
        # traininputpredictions3 = sess.run(y_conv3,feed_dict={x3:batch_xs, keep_prob3:1.0})
        # traininputpredictions4 = sess.run(y_conv4,feed_dict={x4:batch_xs, keep_prob4:1.0})
        # traininputpredictions5 = sess.run(y_conv5,feed_dict={x5:batch_xs, keep_prob5:1.0})
        #
        # level2traininput_temp = tf.concat(
        #     [traininputpredictions1, traininputpredictions2, traininputpredictions3, traininputpredictions4,
        #      traininputpredictions5], 1).eval()
        """ old tuning, not a good approach
        if i<=2000:
            summary, _ = sess.run([merged, level2train_step],feed_dict={level2traininput: level2traininput_temp, level2y_: batch_ys,learning_rate:0.5})
        if i<=10000 and i>2000:
            summary, _ = sess.run([merged, level2train_step],
                                  feed_dict={level2traininput: level2traininput_temp, level2y_: batch_ys,learning_rate:0.005})
        if i<=20000 and i>10000:
            summary, _ = sess.run([merged, level2train_step],
                                  feed_dict={level2traininput: level2traininput_temp, level2y_: batch_ys,learning_rate:0.001})"""
        summary, _=sess.run([merged, level2train_step], feed_dict={level2traininput: batch_xs, level2y_: batch_ys})
        trainwriter.add_summary(summary, i)
        #sess.run(level2train_step)
    end = int(round(time.time() * 1000))
    print("Time for building convnet: ")
    print(end - start)
    # Evaluate the whole model

    # testinputlevel1=tf.constant(test_images)

    testinputpredictions1 = sess.run(y_conv1,feed_dict={x1:mnist.test.images, keep_prob1:1.0})
    testinputpredictions2 = sess.run(y_conv2,feed_dict={x2:mnist.test.images, keep_prob2:1.0})
    testinputpredictions3 = sess.run(y_conv3,feed_dict={x3:mnist.test.images, keep_prob3:1.0})
    testinputpredictions4 = sess.run(y_conv4,feed_dict={x4:mnist.test.images, keep_prob4:1.0})
    testinputpredictions5 = sess.run(y_conv5,feed_dict={x5:mnist.test.images, keep_prob5:1.0})

    level2testinput=tf.concat([testinputpredictions1,testinputpredictions2,testinputpredictions3,testinputpredictions4,testinputpredictions5],1)
    print(level2testinput)

    test_predicted=tf.nn.softmax(tf.matmul(level2testinput, level2W) + level2b, name='test_predicted')
    correct_prediction = tf.equal(tf.argmax(test_predicted, 1), tf.argmax(tf.constant(test_labels), 1))

    # accuracy op
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accu = sess.run(accuracy)
    print(accu)

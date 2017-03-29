# level1 output(without raw input data) to level2
# changed code for tensorflow 1.0
# Tuning global model: learning rate and optimizer
# added GPU lines
import os
import tarfile
import gzip
import tensorflow as tf
import numpy
from tensorflow.contrib.session_bundle import exporter

tf.logging.set_verbosity(tf.logging.DEBUG)
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.InteractiveSession(config=config)
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_images=mnist.train._images
train_labels=mnist.train._labels
print(train_images.shape)
# print(train_images)
# print("Train Labels:")
# print(train_labels)


test_images=mnist.test._images
test_labels=mnist.test._labels
print(test_images.shape)
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


with tf.device('/gpu:0'):
    def local_model(class1, class2):
        train_images_class1 = []
        train_images_class2 = []
        train_labels_originals_class1=[]
        train_labels_originals_class2 = []
        j = 0
        for i in train_labels:
            if (i[class1] == 1):
                train_images_class1.append(train_images[j])
                train_labels_originals_class1.append(i)
            if (i[class2] == 1):
                train_images_class2.append(train_images[j])
                train_labels_originals_class2.append(i)
            j = j + 1


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


        W_temp = tf.Variable(tf.truncated_normal([784, 2], stddev = 0.1), name='W')
        b_temp = tf.Variable(tf.truncated_normal([2]), name='b')

        y = tf.nn.softmax(tf.matmul(tf.concat([class1input, class2input], 0), W_temp) + b_temp, name='y')
        sess.run(tf.variables_initializer([class1input, class2input, W_temp, b_temp]))
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=tf.concat([class1label, class2label], 0)))
        # inverse time decay for learning rate
        global_step = tf.Variable(0, trainable=False)
        sess.run(tf.variables_initializer([global_step]))
        starter_learning_rate = 0.1
        k = 0.5
        learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, decay_rate=k, decay_steps=20)

        train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        for i in range(200):
            global_step = i
            sess.run(train_step)

        # Evaluate individual model
        # test images
        test_images_class1 = []
        test_images_class2 = []

        j = 0
        for i in test_labels:
            if (i[class1] == 1):
                test_images_class1.append(test_images[j])
            if (i[class2] == 1):
                test_images_class2.append(test_images[j])
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
        # compare predicted label and actual label
        test_predicted=tf.nn.softmax(tf.matmul(tf.concat([class1test,class2test], 0), W_temp) + b_temp, name='test_predicted')

        correct_prediction = tf.equal(tf.argmax(test_predicted, 1), tf.argmax(tf.concat([class1testlabel,class2testlabel], 0), 1))

        # accuracy op
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        accu = sess.run(accuracy)
        print(accu)

        y_actual=tf.concat([tf.constant(numpy.array(train_labels_originals_class1)), tf.constant(numpy.array(train_labels_originals_class2))], 0)
        #return W_temp, b_temp, class1input, class2input, class1test, class2test

        # return Weight array, biases array, input image vector, predictions of individual model, input train label
        return W_temp,b_temp, tf.concat([class1input,class2input], 0), y, y_actual

    #print(W1,b1,traininput1,trainpredictions1,trainlabels1)
    #print(W1.eval(),b1.eval())
    W1,b1, traininput1,trainpredictions1,trainlabels1=local_model(0,1)
    W2,b2, traininput2,trainpredictions2,trainlabels2=local_model(6,8)
    W3,b3, traininput3,trainpredictions3,trainlabels3=local_model(3,4)
    W4,b4, traininput4,trainpredictions4,trainlabels4=local_model(5,7)
    W5,b5, traininput5,trainpredictions5,trainlabels5=local_model(2,9)

    # level2
    localmodel_input, localmodel_labels = mnist.train.images, mnist.train.labels
    traininputlevel1=tf.constant(localmodel_input)
    traininputpredictions1=tf.nn.softmax(tf.matmul(traininputlevel1, W1) + b1)
    traininputpredictions2=tf.nn.softmax(tf.matmul(traininputlevel1, W2) + b2)
    traininputpredictions3=tf.nn.softmax(tf.matmul(traininputlevel1, W3) + b3)
    traininputpredictions4=tf.nn.softmax(tf.matmul(traininputlevel1, W4) + b4)
    traininputpredictions5=tf.nn.softmax(tf.matmul(traininputlevel1, W5) + b5)

    level2traininput_temp=tf.concat([traininputpredictions1,traininputpredictions2,traininputpredictions3,traininputpredictions4,traininputpredictions5],1).eval()

    level2traininput = tf.placeholder(tf.float32, [None, 10])

    level2W = tf.Variable(tf.truncated_normal([10, 10],stddev=0.01),name='W')
    level2b = tf.Variable(tf.zeros([10]),name='b')
    sess.run(tf.variables_initializer([level2W, level2b]))

    level2y = tf.matmul(level2traininput, level2W) + level2b
    level2y_ = tf.placeholder(tf.float32, [None, 10],name='level2y_')

    # level2cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=level2y, labels=level2y_))
    level2cross_entropy = tf.reduce_sum(tf.cast(tf.argmax(level2y_, 1)-tf.argmax(level2y, 1),tf.float32))
    tf.summary.scalar('cross_entropy', level2cross_entropy)


    # exponential decay for learning rate
    # global_step = tf.Variable(0, trainable=False)
    # starter_learning_rate = 0.1
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
    #                                            20000, 0.96, staircase=True)
    # inverse time decay for learning rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.5
    k = 0.5
    learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, decay_rate=k,decay_steps=10000)


    tf.summary.scalar('learning_rate', learning_rate)
    # Passing global_step to minimize() will increment it at each step.
    #level2train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(level2cross_entropy, global_step = global_step)
    level2train_step = tf.train.GradientDescentOptimizer(0.1).minimize(level2cross_entropy)


    #learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    # level2train_step = tf.train.AdamOptimizer(learning_rate).minimize(level2cross_entropy)
    #level2train_step = tf.train.GradientDescentOptimizer(0.5).minimize(level2cross_entropy)
    merged = tf.summary.merge_all()
    trainwriter = tf.summary.FileWriter('data/logs', sess.graph)
    sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()))
    for i in range(100000):
        global_step=i
        # batch_xs, batch_ys = mnist.train.next_batch(100)

        batch_xs, batch_ys = next_batch(level2traininput_temp, localmodel_labels, 1000)

        if (i%10000)==0:
            print(i)
        summary, _ = sess.run([merged, level2train_step],
                              feed_dict={level2traininput: batch_xs, level2y_: batch_ys})
        # if i<=4000:
        #     summary, _ = sess.run([merged, level2train_step],feed_dict={traininputlevel1: batch_xs, level2y_: batch_ys,learning_rate:0.1})
        # if i<=100000 and i>4000:
        #     summary, _ = sess.run([merged, level2train_step],
        #                           feed_dict={traininputlevel1: batch_xs, level2y_: batch_ys,learning_rate:0.1})
        # if i<=1000000 and i>100000:
        #     summary, _ = sess.run([merged, level2train_step],
        #                           feed_dict={traininputlevel1: batch_xs, level2y_: batch_ys,learning_rate:0.01})
        # #summary, _=sess.run([merged, level2train_step], feed_dict={traininputlevel1: batch_xs, level2y_: batch_ys})
        trainwriter.add_summary(summary, i)

    # Evaluate the whole model
    testinputlevel1=tf.constant(test_images)

    testinputpredictions1=tf.nn.softmax(tf.matmul(testinputlevel1, W1) + b1)
    testinputpredictions2=tf.nn.softmax(tf.matmul(testinputlevel1, W2) + b2)
    testinputpredictions3=tf.nn.softmax(tf.matmul(testinputlevel1, W3) + b3)
    testinputpredictions4=tf.nn.softmax(tf.matmul(testinputlevel1, W4) + b4)
    testinputpredictions5=tf.nn.softmax(tf.matmul(testinputlevel1, W5) + b5)

    level2testinput=tf.concat([testinputpredictions1,testinputpredictions2,testinputpredictions3,testinputpredictions4,testinputpredictions5],1)
    print(level2testinput)

    test_predicted=tf.matmul(level2testinput, level2W) + level2b
    correct_prediction = tf.equal(tf.argmax(test_predicted, 1), tf.argmax(tf.constant(test_labels), 1))

    # accuracy op
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accu = sess.run(accuracy)
    print(accu)

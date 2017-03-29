# trying with just 2 local models
#level1->level2->fullyconnected layer->softmaxlayer
# so, an extra hidden layer in global model
# level1 output(without raw input data) to level2
# changed code for tensorflow 1.0
# Tuning global model: learning rate and optimizer
# added GPU lines
import os
import tarfile
import gzip
import tensorflow as tf
import numpy
import time
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
    numofexp = images.shape[0]
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
    def local_model(class1, class2,class3,class4,class5):
        train_images_class1 = []
        train_images_class2 = []
        train_images_class3 = []
        train_images_class4 = []
        train_images_class5 = []
        train_labels_originals_class1 = []
        train_labels_originals_class2 = []
        train_labels_originals_class3 = []
        train_labels_originals_class4 = []
        train_labels_originals_class5 = []
        j = 0
        for i in train_labels:
            if (i[class1] == 1):
                train_images_class1.append(train_images[j])
                train_labels_originals_class1.append(i)
            if (i[class2] == 1):
                train_images_class2.append(train_images[j])
                train_labels_originals_class2.append(i)
            if (i[class3] == 1):
                train_images_class3.append(train_images[j])
                train_labels_originals_class3.append(i)
            if (i[class4] == 1):
                train_images_class4.append(train_images[j])
                train_labels_originals_class4.append(i)
            if (i[class5] == 1):
                train_images_class5.append(train_images[j])
                train_labels_originals_class5.append(i)
            j = j + 1


        class1input = tf.Variable(numpy.array(train_images_class1), name='class1input', dtype="float32")
        class2input = tf.Variable(numpy.array(train_images_class2), name='class2input', dtype="float32")
        class3input = tf.Variable(numpy.array(train_images_class3), name='class3input', dtype="float32")
        class4input = tf.Variable(numpy.array(train_images_class4), name='class4input', dtype="float32")
        class5input = tf.Variable(numpy.array(train_images_class5), name='class5input', dtype="float32")
        #print(class1input)

        # building labels
        class1label_1 = tf.Variable(tf.ones([class1input.get_shape()[0], 1]), name='class1label')
        class1label_2 = tf.Variable(tf.zeros([class1input.get_shape()[0], 1]), name='class1label')
        class1label_3 = tf.Variable(tf.zeros([class1input.get_shape()[0], 1]), name='class1label')
        class1label_4 = tf.Variable(tf.zeros([class1input.get_shape()[0], 1]), name='class1label')
        class1label_5 = tf.Variable(tf.zeros([class1input.get_shape()[0], 1]), name='class1label')
        sess.run(tf.variables_initializer([class1label_1, class1label_2,class1label_3,class1label_4,class1label_5]))
        class1label = tf.concat( [class1label_1, class1label_2,class1label_3,class1label_4,class1label_5],1)

        class2label_1 = tf.Variable(tf.zeros([class2input.get_shape()[0], 1]), name='class2label')
        class2label_2 = tf.Variable(tf.ones([class2input.get_shape()[0], 1]), name='class2label')
        class2label_3 = tf.Variable(tf.zeros([class2input.get_shape()[0], 1]), name='class2label')
        class2label_4 = tf.Variable(tf.zeros([class2input.get_shape()[0], 1]), name='class2label')
        class2label_5 = tf.Variable(tf.zeros([class2input.get_shape()[0], 1]), name='class2label')
        sess.run(tf.variables_initializer([class2label_1, class2label_2,class2label_3,class2label_4,class2label_5]))
        class2label = tf.concat( [class2label_1, class2label_2,class2label_3,class2label_4,class2label_5],1)

        class3label_1 = tf.Variable(tf.zeros([class3input.get_shape()[0], 1]), name='class3label')
        class3label_2 = tf.Variable(tf.zeros([class3input.get_shape()[0], 1]), name='class3label')
        class3label_3 = tf.Variable(tf.ones([class3input.get_shape()[0], 1]), name='class3label')
        class3label_4 = tf.Variable(tf.zeros([class3input.get_shape()[0], 1]), name='class3label')
        class3label_5 = tf.Variable(tf.zeros([class3input.get_shape()[0], 1]), name='class3label')
        sess.run(tf.variables_initializer([class3label_1, class3label_2, class3label_3, class3label_4, class3label_5]))
        class3label = tf.concat([class3label_1, class3label_2, class3label_3, class3label_4, class3label_5], 1)

        class4label_1 = tf.Variable(tf.zeros([class4input.get_shape()[0], 1]), name='class4label')
        class4label_2 = tf.Variable(tf.zeros([class4input.get_shape()[0], 1]), name='class4label')
        class4label_3 = tf.Variable(tf.zeros([class4input.get_shape()[0], 1]), name='class4label')
        class4label_4 = tf.Variable(tf.ones([class4input.get_shape()[0], 1]), name='class4label')
        class4label_5 = tf.Variable(tf.zeros([class4input.get_shape()[0], 1]), name='class4label')
        sess.run(tf.variables_initializer([class4label_1, class4label_2, class4label_3, class4label_4, class4label_5]))
        class4label = tf.concat([class4label_1, class4label_2, class4label_3, class4label_4, class4label_5], 1)

        class5label_1 = tf.Variable(tf.zeros([class5input.get_shape()[0], 1]), name='class5label')
        class5label_2 = tf.Variable(tf.zeros([class5input.get_shape()[0], 1]), name='class5label')
        class5label_3 = tf.Variable(tf.zeros([class5input.get_shape()[0], 1]), name='class5label')
        class5label_4 = tf.Variable(tf.zeros([class5input.get_shape()[0], 1]), name='class5label')
        class5label_5 = tf.Variable(tf.ones([class5input.get_shape()[0], 1]), name='class5label')
        sess.run(tf.variables_initializer([class5label_1, class5label_2, class5label_3, class5label_4, class5label_5]))
        class5label = tf.concat([class5label_1, class5label_2, class5label_3, class5label_4, class5label_5], 1)


        # building labels approach2
        # original labels
        # original_labels=tf.Variable(train_labels_originals_class1)
        # local_labels=original_labels


        # x=tf.placeholder(tf.float32, [None, 784],name='x')
        x=tf.concat([class1input, class2input,class3input,class4input,class5input], 0)

        W_temp = tf.Variable(tf.truncated_normal([784, 5], stddev=0.1), name='W')
        b_temp = tf.Variable(tf.zeros([5]), name='b')

        y = tf.nn.softmax(tf.matmul(x, W_temp) + b_temp, name='y')
        sess.run(tf.variables_initializer([class1input, class2input,class3input,class4input,class5input, W_temp, b_temp]))

        #y_=tf.placeholder(tf.float32, [None, 5], name='y_')
        y_=tf.concat([class1label, class2label, class3label, class4label, class5label], 0)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        # tf.summary.scalar('cross_entropy', cross_entropy)


        # inverse time decay for learning rate
        global_step = tf.Variable(0, trainable=False)
        sess.run(tf.variables_initializer([global_step]))
        starter_learning_rate = 0.01
        k = 0.5
        learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, decay_rate=k, decay_steps=10)

        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, global_step=global_step)

        # merged = tf.summary.merge_all()
        # trainwriter = tf.summary.FileWriter('data/logs/train', sess.graph)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        # for all data
        localmodel_images=tf.cast(tf.concat([class1input, class2input,class3input,class4input,class5input], 0),tf.float32).eval()
        # for all data
        localmodel_labels=tf.cast(tf.concat([class1label, class2label, class3label, class4label, class5label], 0),tf.float32).eval()

        # print(localmodel_images.shape[0])
        for i in range(800):
            # print(i)
            global_step = i

            batch_xs,batch_ys=localmodel_images, localmodel_labels
            # print(batch_xs.shape)
            # print(batch_xs.dtype)
            # print(batch_ys.dtype)
            # print(batch_ys.shape)

            # summary2, _ =sess.run([merged,train_step],feed_dict={x:batch_xs,y_:batch_ys})
            _ =sess.run([train_step])
            #trainwriter.add_summary(summary2,i)
            # print(cross_entropy.eval())

        # Evaluate individual model
        # test images
        test_images_class1 = []
        test_images_class2 = []
        test_images_class3 = []
        test_images_class4 = []
        test_images_class5 = []
        test_testlabels_originals_class1=[]
        test_testlabels_originals_class2 = []
        test_testlabels_originals_class3 = []
        test_testlabels_originals_class4 = []
        test_testlabels_originals_class5 = []
        j = 0
        for i in test_labels:
            if (i[class1] == 1):
                test_images_class1.append(test_images[j])
                test_testlabels_originals_class1.append(i)
            if (i[class2] == 1):
                test_images_class2.append(test_images[j])
                test_testlabels_originals_class2.append(i)
            if (i[class3] == 1):
                test_images_class3.append(test_images[j])
                test_testlabels_originals_class3.append(i)
            if (i[class4] == 1):
                test_images_class4.append(test_images[j])
                test_testlabels_originals_class4.append(i)
            if (i[class5] == 1):
                test_images_class5.append(test_images[j])
                test_testlabels_originals_class5.append(i)
            j = j + 1


        class1test = tf.constant(numpy.array(test_images_class1), name='class1test', dtype="float32")
        class2test = tf.constant(numpy.array(test_images_class2), name='class2test', dtype="float32")
        class3test = tf.constant(numpy.array(test_images_class3), name='class3test', dtype="float32")
        class4test = tf.constant(numpy.array(test_images_class4), name='class4test', dtype="float32")
        class5test = tf.constant(numpy.array(test_images_class5), name='class5test', dtype="float32")
        #print(class1test)

        # building testlabels
        class1testlabel_1 = tf.Variable(tf.ones([class1test.get_shape()[0], 1]), name='class1testlabel')
        class1testlabel_2 = tf.Variable(tf.zeros([class1test.get_shape()[0], 1]), name='class1testlabel')
        class1testlabel_3 = tf.Variable(tf.zeros([class1test.get_shape()[0], 1]), name='class1testlabel')
        class1testlabel_4 = tf.Variable(tf.zeros([class1test.get_shape()[0], 1]), name='class1testlabel')
        class1testlabel_5 = tf.Variable(tf.zeros([class1test.get_shape()[0], 1]), name='class1testlabel')
        sess.run(tf.variables_initializer([class1testlabel_1, class1testlabel_2,class1testlabel_3,class1testlabel_4,class1testlabel_5]))
        class1testlabel = tf.concat( [class1testlabel_1, class1testlabel_2,class1testlabel_3,class1testlabel_4,class1testlabel_5],1)

        class2testlabel_1 = tf.Variable(tf.zeros([class2test.get_shape()[0], 1]), name='class2testlabel')
        class2testlabel_2 = tf.Variable(tf.ones([class2test.get_shape()[0], 1]), name='class2testlabel')
        class2testlabel_3 = tf.Variable(tf.zeros([class2test.get_shape()[0], 1]), name='class2testlabel')
        class2testlabel_4 = tf.Variable(tf.zeros([class2test.get_shape()[0], 1]), name='class2testlabel')
        class2testlabel_5 = tf.Variable(tf.zeros([class2test.get_shape()[0], 1]), name='class2testlabel')
        sess.run(tf.variables_initializer([class2testlabel_1, class2testlabel_2,class2testlabel_3,class2testlabel_4,class2testlabel_5]))
        class2testlabel = tf.concat( [class2testlabel_1, class2testlabel_2,class2testlabel_3,class2testlabel_4,class2testlabel_5],1)

        class3testlabel_1 = tf.Variable(tf.zeros([class3test.get_shape()[0], 1]), name='class3testlabel')
        class3testlabel_2 = tf.Variable(tf.zeros([class3test.get_shape()[0], 1]), name='class3testlabel')
        class3testlabel_3 = tf.Variable(tf.ones([class3test.get_shape()[0], 1]), name='class3testlabel')
        class3testlabel_4 = tf.Variable(tf.zeros([class3test.get_shape()[0], 1]), name='class3testlabel')
        class3testlabel_5 = tf.Variable(tf.zeros([class3test.get_shape()[0], 1]), name='class3testlabel')
        sess.run(tf.variables_initializer([class3testlabel_1, class3testlabel_2, class3testlabel_3, class3testlabel_4, class3testlabel_5]))
        class3testlabel = tf.concat([class3testlabel_1, class3testlabel_2, class3testlabel_3, class3testlabel_4, class3testlabel_5], 1)

        class4testlabel_1 = tf.Variable(tf.zeros([class4test.get_shape()[0], 1]), name='class4testlabel')
        class4testlabel_2 = tf.Variable(tf.zeros([class4test.get_shape()[0], 1]), name='class4testlabel')
        class4testlabel_3 = tf.Variable(tf.zeros([class4test.get_shape()[0], 1]), name='class4testlabel')
        class4testlabel_4 = tf.Variable(tf.ones([class4test.get_shape()[0], 1]), name='class4testlabel')
        class4testlabel_5 = tf.Variable(tf.zeros([class4test.get_shape()[0], 1]), name='class4testlabel')
        sess.run(tf.variables_initializer([class4testlabel_1, class4testlabel_2, class4testlabel_3, class4testlabel_4, class4testlabel_5]))
        class4testlabel = tf.concat([class4testlabel_1, class4testlabel_2, class4testlabel_3, class4testlabel_4, class4testlabel_5], 1)

        class5testlabel_1 = tf.Variable(tf.zeros([class5test.get_shape()[0], 1]), name='class5testlabel')
        class5testlabel_2 = tf.Variable(tf.zeros([class5test.get_shape()[0], 1]), name='class5testlabel')
        class5testlabel_3 = tf.Variable(tf.zeros([class5test.get_shape()[0], 1]), name='class5testlabel')
        class5testlabel_4 = tf.Variable(tf.zeros([class5test.get_shape()[0], 1]), name='class5testlabel')
        class5testlabel_5 = tf.Variable(tf.ones([class5test.get_shape()[0], 1]), name='class5testlabel')
        sess.run(tf.variables_initializer([class5testlabel_1, class5testlabel_2, class5testlabel_3, class5testlabel_4, class5testlabel_5]))
        class5testlabel = tf.concat([class5testlabel_1, class5testlabel_2, class5testlabel_3, class5testlabel_4, class5testlabel_5], 1)
        # compare predicted label and actual label
        test_predicted=tf.nn.softmax(tf.matmul(tf.concat([class1test,class2test,class3test,class4test,class5test], 0), W_temp) + b_temp, name='test_predicted')

        correct_prediction = tf.equal(tf.argmax(test_predicted, 1), tf.argmax(tf.concat([class1testlabel,class2testlabel,class3testlabel,class4testlabel,class5testlabel], 0), 1))

        # accuracy op
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        accu = sess.run(accuracy)
        print("local model accuracy: ")
        print(accu)

        y_actual=tf.concat([tf.constant(numpy.array(train_labels_originals_class1)),
                            tf.constant(numpy.array(train_labels_originals_class2)),
                            tf.constant(numpy.array(train_labels_originals_class3)),
                            tf.constant(numpy.array(train_labels_originals_class4)),
                            tf.constant(numpy.array(train_labels_originals_class5))], 0)
        #return W_temp, b_temp, class1input, class2input, class1test, class2test

        # return Weight array, biases array, input image vector, predictions of individual model, input train label
        # return tf.constant(W_temp.eval()), tf.constant(b_temp.eval()), tf.concat([class1input,class2input,class3input,class4input,class5input], 0), y, y_actual
        return tf.constant(W_temp.eval()), tf.constant(b_temp.eval())
    #print(W1,b1,traininput1,trainpredictions1,trainlabels1)
    #print(W1.eval(),b1.eval())
    W1,b1=local_model(0,1,2,3,4)
    W2,b2=local_model(5,6,7,8,9)
    # print(W1.eval())



    # level2 -- hidden layer1
    traininputlevel1=tf.placeholder(tf.float32, [None, 784])
    traininputpredictions1=tf.nn.softmax(tf.matmul(traininputlevel1, W1) + b1)
    traininputpredictions2=tf.nn.softmax(tf.matmul(traininputlevel1, W2) + b2)


    level2traininput=tf.concat([traininputpredictions1,traininputpredictions2],1)
    # print(level2traininput)

    level2W = tf.Variable(tf.truncated_normal([10, 100],stddev=0.1),name='W')
    level2b = tf.Variable(tf.truncated_normal([100]),name='b')
    sess.run(tf.variables_initializer([level2W, level2b]))

    # hidden layer2
    hidden2traininput = tf.nn.relu(tf.matmul(level2traininput, level2W) + level2b)
    hidden2W = tf.Variable(tf.truncated_normal([100, 100], stddev=0.1), name='W')
    hidden2b = tf.Variable(tf.truncated_normal([100]), name='b')


    # #hidden layer3
    # hidden3traininput = tf.nn.relu(tf.matmul(hidden2traininput, hidden2W) + hidden2b)
    # hidden3W = tf.Variable(tf.truncated_normal([100, 100], stddev=0.1), name='W')
    # hidden3b = tf.Variable(tf.truncated_normal([100]), name='b')


    level3traininput = tf.nn.relu(tf.matmul(hidden2traininput, hidden2W) + hidden2b)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(level3traininput, keep_prob)

    # level3 or readout layer

    level3W = tf.Variable(tf.truncated_normal([100, 10],stddev=0.1), name='W')
    level3b = tf.Variable(tf.truncated_normal([10]), name='b')
    # sess.run(tf.variables_initializer([level3W, level3b]))

    # level3y = tf.matmul(level3traininput, level3W) + level3b

    level3y = tf.nn.softmax(tf.matmul(h_fc1_drop, level3W) + level3b, name='level3y')
    level3y_ = tf.placeholder(tf.float32, [None, 10], name='level3y_')

    level3cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=level3y, labels=level3y_))
    tf.summary.scalar('cross_entropy', level3cross_entropy)


    # exponential decay for learning rate
    # global_step = tf.Variable(0, trainable=False)
    # starter_learning_rate = 0.1
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
    #                                            20000, 0.96, staircase=True)

    #inverse time decay gave good results in less time
    # inverse time decay for learning rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    k = 0.5
    learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, decay_rate=k,decay_steps=100)

    # # piecewise constant function
    # global_step = tf.Variable(0, trainable=False)
    # boundaries = [30000, 50000,80000,140000,170000]
    # values = [0.01, 0.005, 0.0005,1e-15,1e-16,1e-18]
    # learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)



    tf.summary.scalar('learning_rate', learning_rate)
    # Passing global_step to minimize() will increment it at each step.
    level3train_step = (tf.train.RMSPropOptimizer(learning_rate).minimize(level3cross_entropy, global_step= global_step))

    #learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    # level2train_step = tf.train.AdamOptimizer(learning_rate).minimize(level2cross_entropy)
    #level2train_step = tf.train.GradientDescentOptimizer(0.5).minimize(level2cross_entropy)

    # to compute validation set accuracy (to observe over fitting)
    validation_input=mnist.validation.images
    validation_labels=mnist.validation.labels

    prediction = tf.argmax(level3y, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(level3y_, 1))
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)




    merged = tf.summary.merge_all()
    trainwriter = tf.summary.FileWriter('data/logs/train', sess.graph)
    validationwriter = tf.summary.FileWriter('data/logs/validation', sess.graph)
    sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()))
    start = int(round(time.time() * 1000))
    for i in range(200000):
        global_step=i
        batch_xs, batch_ys = mnist.train.next_batch(1000)

        if (i%1000)==0:
            print(i)


        summary, _ = sess.run([merged, level3train_step],
                              feed_dict={traininputlevel1: batch_xs, level3y_: batch_ys,keep_prob:0.5})
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

        # validation
        summary,_=sess.run([merged, level3y],feed_dict={traininputlevel1: validation_input, level3y_: validation_labels,keep_prob:1})
        validationwriter.add_summary(summary, i)

    # Evaluate the whole model
    testinputlevel1=tf.constant(test_images)

    testinputpredictions1=tf.nn.softmax(tf.matmul(testinputlevel1, W1) + b1)
    testinputpredictions2=tf.nn.softmax(tf.matmul(testinputlevel1, W2) + b2)


    level2testinput=tf.concat([testinputpredictions1,testinputpredictions2],1)

    test_predicted=tf.nn.softmax(tf.matmul(h_fc1_drop, level3W) + level3b, name='test_predicted')
    test_predictions=sess.run(level3y, feed_dict={traininputlevel1: mnist.test.images, keep_prob: 1.0})

    correct_prediction = tf.equal(tf.argmax(test_predictions, 1), tf.argmax(tf.constant(test_labels), 1))

    # accuracy op
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accu = sess.run(accuracy)
    print(accu*100)
    end = int(round(time.time() * 1000))
    print("Time for building 2 layer global model: ")
    print(end - start)

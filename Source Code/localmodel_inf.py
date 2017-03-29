# inference on saved data
import tensorflow as tf
import os.path
import numpy


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess=tf.InteractiveSession()

def get_inference(model_num,input):
    model_dir='data/mnist-model'+str(model_num)
    # print(model_dir)
    # restore the saved model
    new_saver = tf.train.import_meta_graph(model_dir+'.meta')
    new_saver.restore(sess, model_dir)

    """
    # print to see the restored variables
    for v in tf.get_collection('variables'):
        print(v.name)
    print(sess.run(tf.global_variables()))

    # print ops
    for op in sess.graph.get_operations():
        print(op.name)
    """

    x=sess.graph.get_tensor_by_name('x:0')
    # placeholders for test images and labels

    y_conv = sess.graph.get_tensor_by_name('y_conv:0')
    keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
    # print(x)
    # print(y_conv)
    predictions=sess.run(y_conv,feed_dict={x:input,keep_prob:1.0})
    # print(predictions)
    return predictions

#traininputlevel1=mnist.train.images.next_batch(2)
batch_xs, batch_ys = mnist.train.next_batch(50)
print(type(batch_xs))
print(numpy.array(batch_xs))
traininputpredictions1=get_inference(1, batch_xs)

print(isinstance(mnist.train.images, (numpy.ndarray)))
print(mnist.train.images.dtype)

temp=tf.truncated_normal([2,10], stddev=0.1)
print(temp)
print(type(temp.eval()))


















"""Creates a graph from saved GraphDef file and returns a saver."""
"""# Creates graph from saved graph_def.pb.
f=tf.gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb')
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
_ = tf.import_graph_def(graph_def, name='')

for op in sess.graph.get_operations():
    print(op.name)

# We access the input and output nodes
x = sess.graph.get_tensor_by_name('images:0')
y = sess.graph.get_tensor_by_name('scores:0')

# We launch a Session

# Note: we didn't initialize/restore anything, everything is stored in the graph_def
prediction = sess.run(y, feed_dict={x: mnist.test.images })
print(prediction)
"""

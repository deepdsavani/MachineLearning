import tensorflow as tf
import numpy as np

#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("data", one_hot=True)

#images_test = mnist.test.images

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_train shape:", x_test.shape, "y_train shape:", y_test.shape)

IMAGE_PIXELS = x_train.shape[1]*x_train.shape[2]*x_train.shape[3];
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3]);
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3]);

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_train shape:", x_test.shape, "y_train shape:", y_test.shape)

x_train=x_train/255.0;
x_test=x_test/255.0;

y_Train = [];
y_Test = [];

for l in y_train:
    t = np.zeros(10);
    t[l]=1;
    y_Train.append(t);

for l in y_test:
    t = np.zeros(10);
    t[l]=1;
    y_Test.append(t);

y_train = np.array(y_Train, dtype=np.float32);
y_test = np.array(y_Test, dtype=np.float32);


n_classes = 10
batch_size = 128

# Matrix -> height x width
# height = None, width = 32x32

x = tf.placeholder('float',[None, 32*32*3])
y = tf.placeholder('float')


def conv2d(x, W):
    
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
def maxpool2d(x):
    
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x) :
    
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,32])),     # 5x5 convolution 1 input 32 features
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([8*8*64,1024])),
               'out':tf.Variable(tf.random_normal([1024,n_classes]))}
                      
    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),     # 5x5 convolution 1 input 32 features
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_fc':tf.Variable(tf.random_normal([1024])),
              'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1,32,32,3])
    conv1 = tf.add(conv2d(x, weights['W_conv1']),biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    conv2 = (tf.add(conv2d(conv1, weights['W_conv2']),biases['b_conv2']))
    conv2 = maxpool2d(conv2)
    
    fc = tf.reshape(conv2, [-1, 8*8*64])
    fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['W_fc']),biases['b_fc']))
    
    output = tf.add(tf.matmul(fc,weights['out']),biases['out'])
    
    return output
    
    
prediction = convolutional_neural_network(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

optimizer = tf.train.AdamOptimizer().minimize(cost)

n_epochs = 20	 # cycles of feed forward + backprop

cur = 0

def get_next_epoch_data():
    
    out_x = x_train[cur:cur+batch_size,:]
    out_y = y_train[cur:cur+batch_size]
    return out_x, out_y

with tf.Session() as sess :
    
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epochs) :
        
        epoch_loss = 0
        cur = 0
        
        for _ in range(int(len(x_train)/batch_size) ) :
            
            epoch_x, epoch_y = get_next_epoch_data()
            cur += batch_size
            _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
            epoch_loss += c
            
        print ('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)
    
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    
    
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    accs = {}
    
    for i in range(int(len(x_test)/batch_size) ) :
        
        accs[i] = accuracy.eval({x:x_test[(i*batch_size):((i+1)*batch_size)], y:y_test[(i*batch_size):((i+1)*batch_size)]})
    
    tot = 0.0
    
    for i in accs :
        
        tot+=accs[i]
        
    print ("Accuracy:", float(tot/((len(x_test)/batch_size))) )
                
#train_neural_network(x)
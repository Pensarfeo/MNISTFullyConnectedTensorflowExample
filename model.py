import tensorflow as tf
SEED = 66478

#------------------------------------------------------------------
# A function to create fc layers
#------------------------------------------------------------------
def fc(input, cout, name='fc', data_type = tf.float32):
  cin = input.shape.as_list()[-1]

  #  define variables to be used as weights and biaces
  with tf.name_scope(name):
    weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal(
        [cin, cout],
        stddev=0.1,
        seed=SEED,
        dtype=data_type
      ),
      name='W'
    )
    biases = tf.Variable(tf.constant(0.1, shape=[cout], dtype=data_type), name='B')

    # Combine input and output to get cell output
    output = tf.matmul(input, weights) + biases

    #  Save info for tf board
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("biases", biases)
    tf.summary.histogram("activations", output)

    return [output, weights, biases]


def net(data, train=False, data_type = tf.float16):
  data_shape = data.get_shape().as_list()
  data_to_fc = [data_shape[0], data_shape[1] * data_shape[2] * data_shape[3]]
  reshape = tf.reshape(data, data_to_fc)
  
  fc1, fc1_weights, fc1_biases = fc(reshape, 512, 'fc1', data_type)
  fc1 = tf.nn.relu(fc1)
  fc2, fc2_weights, fc2_biases = fc(fc1, 512//2, 'fc2', data_type)
  fc2 = tf.nn.relu(fc2)
  fc3, fc3_weights, fc3_biases = fc(fc2, 512//4, 'fc3', data_type)
  fc3 = tf.nn.relu(fc3)
  fc4, fc4_weights, fc4_biases = fc(fc3, 10, 'fc4', data_type)
  # NOTICE WE ARE NOT DOING RELU CAUSE (1) our cross entropy function does logits (2) relu is not bounded withing 0 and 1

  return [fc4, fc1_weights, fc1_biases, fc2_weights, fc2_biases, fc3_weights, fc3_biases, fc4_weights, fc4_biases]
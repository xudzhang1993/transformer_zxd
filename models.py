import tensorflow as tf
import numpy as np
import math


def position_encoding_init(d_model, max_length):
    pos = np.arange(max_length).reshape([-1,1]) * np.ones(d_model/2).reshape([1, -1])
    i = np.arange(d_model/2).reshape([1, -1]) * np.ones(max_length).reshape([-1, 1])
    sin = np.sin(pos/np.power(10000, (2.0 * i / d_model)))
    cos = np.cos(pos/np.power(10000, ((2.0 *  (i + 1)) / d_model))) # ? 2i or 2i + 1
    encoding = np.zeros([max_length, d_model])
    encoding[:, 0::2] = sin
    encoding[:, 1::2] = cos
    encoding = encoding.astype('float32')
    #encoding = encoding * (d_model ** 0.5) 
    return encoding  # max_length, d_model

def position_layer(encoding, minibatch_length, scope):
    with tf.variable_scope(scope) as scope:
        encoding_tensor = tf.get_variable('encoding', initializer=encoding, trainable=False, dtype=tf.float32)
        encoding_tensor = tf.expand_dims(encoding_tensor[:minibatch_length, :], 0)    # slice the encoding_tensor according to minibatch length, expand dims for broadcasting
        return encoding_tensor
        

def test_position_init():
    print position_encoding_init(512, 256)


def input_layer(encoding, ids, vocab_size, seq_length, d_model, keep_prob, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        # add embedding outputs with position encoding
        minibatch_length = ids.shape[1].value or tf.shape(ids)[1]
        embedding_output = embedding_layer(ids, vocab_size, d_model, "embedding_layer", reuse=reuse)
        position_encoding = position_layer(encoding, minibatch_length, "position_layer") 
        outputs = position_encoding + embedding_output # batch, minibatch_length, d_model

        # mask paded symbol with 0 value
        mask = tf.sequence_mask(seq_length, minibatch_length) # batch, minibatch_length
        mask = tf.expand_dims(mask, 2) # batch, minibatch_length, 1
        outputs = outputs * tf.to_float(mask)

        # embedding dropout
        outputs = tf.nn.dropout(outputs, keep_prob)

        return outputs


def test_input_layer():
    d_model = 512
    max_length = 1024
    mini_length = 256
    vocab_size = 100
    batch = 32
    keep_prob = 0.9
    seq_length = tf.Variable(np.random.randint(0, mini_length, [batch]))
    encoding = position_encoding_init(d_model, max_length)
    ids = tf.Variable(np.random.randint(0, vocab_size, [batch, mini_length])) 
    outputs = input_layer(encoding, ids, vocab_size, mini_length, seq_length, d_model, keep_prob, "input")
    print outputs


def embedding_layer(ids, vocab_size, d_model, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', shape=[vocab_size, d_model], dtype=tf.float32)
        outputs = tf.nn.embedding_lookup(lookup_table, ids)
        outputs = outputs * tf.sqrt(tf.to_float(d_model))
        return outputs
        

def layer_norm(inputs, scope, reuse=False, epsilon=1e-5):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape), name="beta")
        gamma = tf.Variable(tf.ones(params_shape), name="gamma")
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
        return outputs


def multihead_attention(query, key, d_qkv, num_heads, keep_prob, query_seq_length, key_seq_length, scope, decoder_self_attention=False, reuse=False):
    """

    :param query: batch, q_l, depth
    :param key: batch, k_l, depth
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse) as scope:
        d_model = query.shape[-1].value or tf.shape(query)[-1]
        q_l = query.shape[1].value or tf.shape(query)[1] 
        k_l = key.shape[1].value or tf.shape(key)[1]
        # num_units equals to numbers of head * dimension of query vectors
        num_units = num_heads * d_qkv
        # create Q,K,V in a concatenated form
        Q = tf.layers.dense(query, num_units, activation=None, use_bias=False)
        K = tf.layers.dense(key, num_units, activation=None, use_bias=False)
        V = tf.layers.dense(key, num_units, activation=None, use_bias=False)
        # split concatenated Q,K,V into multi heads
        Q_heads = tf.concat(tf.split(Q, num_heads, axis=-1), 0)  # num_heads * batch, q_l, d_qkv
        K_heads = tf.concat(tf.split(K, num_heads, axis=-1), 0)  # num_heads * batch, k_l, d_qkv
        V_heads = tf.concat(tf.split(V, num_heads, axis=-1), 0)  # num_heads * batch, k_l, d_qkv
        score = tf.matmul(Q_heads, K_heads, transpose_b=[0, 2, 1]) / tf.sqrt(
            tf.to_float(d_qkv))  # num_heads * batch, q_l, k_l
        # key_mask mask score with -inf in paded position of key
        key_mask = tf.sequence_mask(key_seq_length, k_l) # batch, k_l
        key_mask = tf.expand_dims(key_mask, 1)
        key_mask = tf.tile(key_mask, [num_heads, q_l, 1]) # num_heads * batch, q_l, k_l
        paddings = tf.ones_like(score) * (-math.pow(2, 32) + 1)
        score = tf.where(tf.equal(key_mask, False), paddings, score)

	# order_mask this mask avoid queries to attend on subsequent keys, which is used for decoder self attention
        # use LinearOperatorTriL to set upper triangular entries 0 (make a matrix lower triangular one), which constrains the causility between the queries and keys
        if decoder_self_attention:
            order_mask = tf.ones_like(score)
            order_mask = tf.contrib.linalg.LinearOperatorTriL(order_mask).to_dense()
            score =  tf.where(tf.equal(order_mask, 0), paddings, score)
        #score = tf.Print(score, [score], summarize=1000) #  print score to check whether mask is working or not
        alignment = tf.nn.softmax(score)
        alignment =tf.nn.dropout(alignment, keep_prob)  # attention dropout by random set attention weight to zero
        attention = tf.matmul(alignment, V_heads)  # num_heads * batch, q_l, d_qkv

        # query_mask mask attention values with 0 in padded position of query
        query_mask = tf.sequence_mask(query_seq_length, q_l) # batch, q_l
        query_mask = tf.expand_dims(query_mask, 2)
        query_mask = tf.tile(query_mask, [num_heads, 1, d_qkv]) # num_heads * batch, q_l, d_qkv
        attention = attention * tf.to_float(query_mask)

        attention = tf.concat(tf.split(attention, num_heads, axis=0), axis=-1)  # batch, q_l, num_units
        WO = tf.get_variable('WO', shape=[num_units, d_model])  # last linear transform on all concat attention vectors
        attention = tf.tensordot(attention, WO, [[2], [0]])  # batch, q_l, d_module
        return attention


def res(inputs, output, scope):
    with tf.variable_scope(scope) as scope:
        output = inputs + output
        return output


def feed_forward(inputs, d_ff, scope, reuse=False):
    d_model = inputs.shape[-1].value or tf.shape(inputs)[-1]
    with tf.variable_scope(scope, reuse=reuse):
        f1 = tf.layers.dense(inputs, d_ff, activation=tf.nn.relu, use_bias=True)
        f2 = tf.layers.dense(f1, d_model, activation=None, use_bias=True)
    return f2


def encoder_block(query, key, d_qkv, d_ff, num_heads, keep_prob, query_seq_length, key_seq_length, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        # multi-head attention + residual + layer normalization
        attention = multihead_attention(query, key, d_qkv, num_heads, keep_prob, query_seq_length=query_seq_length, key_seq_length=key_seq_length, scope="multihead_attention")
        attention = tf.nn.dropout(attention, keep_prob)  # sub layer drop out before added to res and normlize
        attention = res(query, attention, "residual_after_attention")
        attention = layer_norm(attention, "layer_normalization_after_attention")
        # feed-forward + residual + layer normalization
        outputs = feed_forward(attention, d_ff, "feed_forward")
        outputs = tf.nn.dropout(outputs, keep_prob)  # sub layer drop out before added to res and normlize
        outputs = res(attention, outputs, "residual_after_feed_forward")
        outputs = layer_norm(outputs, "layer_normalization_after_feed_forward")

        return outputs


def decoder_block(query_dec, key_enc, key, d_qkv, d_ff, num_heads, keep_prob, query_seq_length, key_seq_length, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse) as scope:
        # multi-head attention + residual + layer normalization
        attention_dec = multihead_attention(query_dec, key, d_qkv, num_heads,
                                            keep_prob, query_seq_length=query_seq_length, key_seq_length=query_seq_length, decoder_self_attention=True, scope="multihead_attention_decoder")
        attention_dec = tf.nn.dropout(attention_dec, keep_prob)  # sub layer drop out before added to res and normlize
        attention_dec = res(query_dec, attention_dec, "residual_after_attention_dec")
        attention_dec = layer_norm(attention_dec, "layer_normalization_after_attention_dec")

        # multi-head attention + residual + layer normalization for encoder's outputs as key
        attention_enc = multihead_attention(attention_dec, key_enc, d_qkv, num_heads, keep_prob, query_seq_length=query_seq_length, key_seq_length=key_seq_length,
                                            scope="multihead_attention_encoder")
        attention_enc = tf.nn.dropout(attention_enc, keep_prob)  # sub layer drop out before added to res and normlize
        attention_enc = res(attention_dec, attention_enc, "residual_after_attention_enc")
        attention_enc = layer_norm(attention_enc, "layer_normalization_after_attention_enc")

        # feed-forward + residual + layer normalization
        outputs = feed_forward(attention_enc, d_ff, "feed_forward")
        outputs = tf.nn.dropout(outputs, keep_prob)  # sub layer drop out before added to res and normlize
        outputs = res(attention_enc, outputs, "residual_after_feed_forward")
        outputs = layer_norm(outputs, "layer_normalization_after_feed_forward")

        return outputs


def test_multihead():
    query = tf.Variable(np.ones([32, 128, 64]), dtype=tf.float32)
    key = tf.Variable(np.ones([32, 128, 64]), dtype=tf.float32)
    d_qkv = 64
    num_heads = 8
    sess = tf.Session()
    attention = multihead_attention(query, key, d_qkv, num_heads)
    sess.run(tf.global_variables_initializer())
    result = sess.run(attention)
    print result, result.shape


def test_encoder_block():
    query_seq_length = tf.Variable(np.ones([2]) * 3, dtype=tf.int32)
    key_seq_length = tf.Variable(np.ones([2]) * 3, dtype=tf.int32)
    query = tf.Variable(np.ones([2, 5, 2]), dtype=tf.float32)
    key = tf.Variable(np.ones([2, 5, 2]), dtype=tf.float32)
    d_qkv = 64
    d_ff = 512
    num_heads = 8
    keep_prob = 0.9
    sess = tf.Session()
    attention = encoder_block(query, key, d_qkv, d_ff, num_heads, keep_prob, query_seq_length=query_seq_length, key_seq_length=key_seq_length, scope="encoder_block")
    sess.run(tf.global_variables_initializer())
    result = sess.run(attention)
    print result, result.shape


def test_decoder_block():
    query_seq_length = tf.Variable(np.ones([2]) * 3, dtype=tf.int32)
    key_seq_length = tf.Variable(np.ones([2]) * 3, dtype=tf.int32)
    query = tf.Variable(np.ones([2, 5, 2]), dtype=tf.float32)
    key = tf.Variable(np.ones([2, 5, 2]), dtype=tf.float32)
    d_qkv = 2
    d_ff = 4
    num_heads = 2
    keep_prob = 0.9
    sess = tf.Session()
    attention = decoder_block(query, query, key, d_qkv, d_ff, num_heads, keep_prob, query_seq_length=query_seq_length, key_seq_length=key_seq_length, scope="decoder_block")
    sess.run(tf.global_variables_initializer())
    result = sess.run(attention)
    print result, result.shape


def label_smoothing(inputs, epsilon = 0.1): 
    ''' 
    Implement label smoothing 
 
    Args: 
        inputs: [Tensor], A 3d tensor with shape of [N, T, V] 
        epsilon: [Float], Smoothing rate 
 
    Return: 
        A tensor after smoothing 
    ''' 


    K = inputs.get_shape().as_list()[-1] 
    return ((1 - epsilon) * inputs) + (epsilon / K) 


if __name__ == "__main__":
    print test_position_init()
   

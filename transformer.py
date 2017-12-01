import tensorflow as tf
import numpy as np
from param import Param
from models import *
from dataset import get_iterator

param = Param()
class Transformer(object):

    def __init__(self):
        self.loss = None
        self.predict = None
        self.global_step = None
        self.init_op = None
        pass

    def build(self, mode):
        # get iterator that can iterate the dataset and batch data in supposed format
        if mode != tf.estimator.ModeKeys.TRAIN and mode !=tf.estimator.Modekeys.INFER:
            raise ValueError("mode must be a key in tf.estimator.ModeKeyS")
        index_table = tf.contrib.lookup.index_table_from_file(param.vocab_file, num_oov_buckets=0, default_value=1) # create index_table to map a string to a integer
        data_file = param.data_file
        self.it_train = get_iterator(data_file, index_table) 

        x, y_in, y_out, x_seq_length, y_seq_length = self.it_train.get_next() # got encoder/decoder input ids and their responese sequence length from iterator
        #x = tf.Print(x, [x, y_in, y_out, x_seq_length, y_seq_length])

        # group the initialize op

        y_l = tf.shape(y_in)[1] # minibatch length for decoder input ids

        # build encoder, decoder input layer by get tokens' embedding and add the position encoding on it
        encoding = position_encoding_init(param.d_model, param.max_length)
        print encoding
        encoder_input = input_layer(encoding, x, param.vocab_size, x_seq_length, param.d_model, param.keep_prob, "input")
        decoder_input = input_layer(encoding, y_in, param.vocab_size, y_seq_length, param.d_model, param.keep_prob, "input", reuse=True) # reuse embedding

        # build encoder blocks, self-attention use encoder_input as both queries and keys
        for i in range(param.num_encoder_blocks): 
            encoder_input = encoder_block(encoder_input, encoder_input, param.d_qkv, param.d_ff, param.num_heads, param.keep_prob, x_seq_length, x_seq_length, "encoder_block_%d" % i) 
        encoder_output = encoder_input
# build decoder blocks, self-attention use decoder_input as queries and keys, vanillia attention use encoder's output as keys,  
        for i in range(param.num_decoder_blocks):
            decoder_input = decoder_block(decoder_input, encoder_output, decoder_input, param.d_qkv, param.d_ff, param.num_heads, param.keep_prob, x_seq_length, y_seq_length, "decoder_block_%d" % i) 
        with tf.variable_scope("last_projection"):
            decoder_output = tf.layers.dense(decoder_input, param.vocab_size)
        with tf.variable_scope("loss"):
            mask = tf.sequence_mask(y_seq_length, y_l) 
            mask = tf.Print(mask, [mask], summarize = 1000)
            if mode == tf.estimator.ModeKeys.TRAIN:
                labels = tf.one_hot(y_out, param.vocab_size)
                labels_smoothed = label_smoothing(labels)
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_smoothed, logits=decoder_output)
                #loss = tf.Print(loss, [loss, labels_smoothed], summarize=1000)
                loss = loss * tf.to_float(mask) # batch, y_l
                self.loss = tf.reduce_sum(loss) / (tf.to_float(tf.reduce_sum(y_seq_length))) # per token loss
                tf.summary.scalar('loss', self.loss)
                self.global_step = tf.Variable(0, name = 'global_step', trainable = False) 
                # optimizer 
                self.optimizer = tf.train.AdamOptimizer(learning_rate = param.learning_rate * 10, beta1 = 0.9, beta2 = 0.98, epsilon = 1e-8) 

                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10)
                self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))
                  

            elif mode == tf.estimator.ModeKeys.INFER:
                mask = tf.expand_dims(mask, 2) # batch, y_l, 1
                pred = tf.arg_max(decoder_output * tf.to_float(mask), -1)[: -1] # batch, the last one along length dimension is the predict token for next position

        self.merged = tf.summary.merge_all()
        self.init_op = tf.group(self.it_train.initializer, tf.global_variables_initializer(), tf.tables_initializer())
        def decode():
            # sequential decoding is only used in infererence, during trainning, the mode decode in a parrallel mode.
            # batched decode should use some fuction like maybe_finished  which paded the finished sample with padded value, and continue to decode the rest of samples until all samples are finished or the max decode step is reached.
            pass



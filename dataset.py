import tensorflow as tf
import numpy as np
from param import Param
import json

param = Param()


def get_iterator(data_file, index_table):

    ds = tf.data.TextLineDataset(data_file)
    
    def split(s):
        sparse = tf.string_split(s, '\t')
        
        x_y = tf.sparse_to_dense(sparse.indices, sparse.dense_shape, sparse.values, default_value=tf.constant("<PAD>"))
        x = tf.reshape(x_y[:, 0], [-1])
        y = tf.reshape(x_y[:, 1], [-1])
        y_in = tf.string_join(["<SOS> ", y])
        y_out = tf.string_join([y, " <EOS>"])
        x = tf.string_split(x, ' ')
        y_in = tf.string_split(y_in, ' ')
        y_out = tf.string_split(y_out, ' ')

        x = tf.sparse_to_dense(x.indices, x.dense_shape, x.values, default_value=tf.constant("<PAD>"))
        y_in = tf.sparse_to_dense(y_in.indices, y_in.dense_shape, y_in.values, default_value=tf.constant("<PAD>"))
        y_out = tf.sparse_to_dense(y_out.indices, y_out.dense_shape, y_out.values, default_value=tf.constant("<PAD>"))
        return x, y_in, y_out

    ds = ds.batch(param.batch_size).prefetch(param.shuffle_buffer_size)
    ds = ds.map(split)
    ds = ds.map(lambda x, y_in, y_out: (index_table.lookup(x), index_table.lookup(y_in), index_table.lookup(y_out)))
    ds = ds.map(lambda x, y_in, y_out: (x, y_in, y_out, tf.reduce_sum(tf.sign(x), 1), tf.reduce_sum(tf.sign(y_in), 1))) # calculte the length of x and y
    ds = ds.repeat()
    ds = ds.shuffle(param.shuffle_buffer_size)
    return ds.make_initializable_iterator()

def tst_get_iterator():
    with open('./data/toy.data', 'r') as f:
        print [f.read()]
    index_table = tf.contrib.lookup.index_table_from_file('./data/vocab_1w.data')
    it = get_iterator('./data/toy.data', index_table)
    sess = tf.Session()
    sess.run(it.initializer)
    sess.run(tf.tables_initializer())
    print sess.run(it.get_next())

if __name__ == "__main__":
    tst_get_iterator()
    
                                                    



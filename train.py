import tensorflow as tf
from param import Param
from transformer import Transformer
import os

def print_variables():
    for x in tf.global_variables():
        print "name: ", x.name, "shape: ", x.shape
    
if __name__ == "__main__":
    with tf.Graph().as_default() as graph:
        param = Param()
        model = Transformer()
        model.build(tf.estimator.ModeKeys.TRAIN)
        print_variables()
        sv = tf.train.Supervisor(logdir="./log/", init_op=model.init_op)
        with sv.managed_session() as sess:
            sess.run(model.it_train.initializer)
            while True:
                if sv.should_stop():
                    break
                loss, _ = sess.run([model.loss, model.train_op])
                print loss
        
    

	

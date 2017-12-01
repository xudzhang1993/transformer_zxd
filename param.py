class Param(object):
    # model param
    max_length = 101
    vocab_size = 10000 
    num_encoder_blocks = 6
    num_decoder_blocks = 6
    num_heads = 8
    d_model = 512
    d_qkv = 64
    d_ff = 2048
    keep_prob = 0.9
    # data param
    data_file = './data/toy.data' 
    vocab_file = './data/vocab_1w.data'
    batch_size = 64
    shuffle_buffer_size = 1024
    # experiment param
    epochs = 1000
    learning_rate = 1e-4
    logdir = "./log/"
    checkpoint = "./check/"

   
   

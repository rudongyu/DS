"""
    multi-layer LSTM RNN with attention mechanism
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

class Seq2seqModel(object):

    def __init__(self
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
                 num_samples=512,   # less than target_vocab_size required
                 forward_only=False,
                 dtype=tf.float32):

        """
            size: number of units in each layer
            max_gradient_norm: gradients will be clipped to maximally this norm.
            batch_size: size of batches used during training
            learning_rate: learning rate to start with
            learning_rate_decay_factor: decay learning rate when needed
            forward_only: select to train or predict
        """

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # for sampled softmax, output projection needed
        # output projection makes sense only when
        # num_samples is less than target_vocab_size
        w_t = tf.get_variable("proj_w", [self.target_vocab_size, size],
                                        dtype=dtype)
        w = tf.transpose(w_t)
        b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
        output_projection = (w, b)
        def sampled_loss(labels, logits):
          labels = tf.reshape(labels, [-1, 1])
          # We need to compute the sampled_softmax_loss using 32bit floats to
          # avoid numerical instabilities.
          local_w_t = tf.cast(w_t, tf.float32)
          local_b = tf.cast(b, tf.float32)
          local_inputs = tf.cast(logits, tf.float32)
          return tf.cast(
              tf.nn.sampled_softmax_loss(
                  weights=local_w_t,
                  biases=local_b,
                  labels=labels,
                  inputs=local_inputs,
                  num_sampled=num_samples,
                  num_classes=self.target_vocab_size),
              dtype)
        softmax_loss_function = sampled_loss

        # create multiple layer RNN here
        def single_cell():
          return tf.contrib.rnn.GRUCell(size)
        if use_lstm:
          def single_cell():
            return tf.contrib.rnn.BasicLSTMCell(size)
        cells = tf.contrib.rnn.MultiRNNCell([single_cell() \
                                            for _ in range(num_layers)])

        # RNN model with embedding and attention
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
          return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
              encoder_inputs,
              decoder_inputs,
              cells,
              num_encoder_symbols=source_vocab_size,
              num_decoder_symbols=target_vocab_size,
              embedding_size=size,
              output_projection=output_projection,
              feed_previous=do_decode,
              dtype=dtype)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
          self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
          self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
          self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                    name="weight{0}".format(i)))

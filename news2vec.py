#!usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import collections
import math
import random
import numpy as np
from six.moves import xrange
import tensorflow as tf
import pandas as pd



class newsfeature2vec:
    def __init__(self, walks, out_dir, batch_size=10, embedding_size=128, skip_window=5, num_skips=10, neg_samples=5, include=True, iter=3000000):
        self.iter = iter
        self.data_index = 0
        if not include:
            self.words = [i for i in walks if ',' in i]
        else:
            self.words = walks
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.neg_samples = neg_samples
        self.build_dataset()
        self.build_dataset_emb()
        self.build_model()

        with tf.Session(graph=self.graph) as session:
            session.run(tf.global_variables_initializer())
            print("Initialized")
            average_loss = 0
            for step in xrange(self.iter):
                batch, labels = self.generate_batch()
                for i in range(self.batch_size):
                    word = self.reverse_dictionary[batch[i]]
                    if ',' in word:
                        inputs = np.array([self.dictionary_emb[i] for i in word.split(',')], dtype=int)
                    else:
                        continue
                        # inputs = np.array(self.dictionary_emb[word], dtype=int)
                    feed_dict = {self.train_inputs: inputs.reshape(1,-1), self.train_labels: labels[i].reshape(1,1)}

                    _, loss_val= session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                    average_loss += loss_val

                if step % 2000 == 0 and step > 0:

                    average_loss /= (2000*self.batch_size)
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0

            final = self.normalized_embeddings.eval()
            pd.DataFrame(final, index=[i[0] for i in self.count_emb]).to_csv(out_dir)

    def build_dataset(self):
        self.count = []
        self.count.extend([list(item) for item in collections.Counter(self.words).most_common()])
        self.vocabulary_size = len(self.count)
        self.dictionary = dict()
        for word, _ in self.count:
            self.dictionary[word] = len(self.dictionary)
        self.data = list()
        del self.count
        for word in self.words:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                index = 0
            self.data.append(index)
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

    def build_dataset_emb(self):
        self.words_emb = []
        for word in [word.split(',') for word in self.words]:
            self.words_emb += word
        self.count_emb = []
        self.count_emb.extend([list(item) for item in collections.Counter(self.words_emb).most_common()])
        self.vocabulary_size_emb = len(self.count_emb)
        self.dictionary_emb = dict()
        for word, _ in self.count_emb:
            self.dictionary_emb[word] = len(self.dictionary_emb)

    # Function to generate a training batch for the skip-gram model.
    def generate_batch(self):
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window
        batch = np.ndarray(shape=[self.batch_size], dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(self.batch_size // self.num_skips):
            target = self.skip_window  # target label at the center of the buffer
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels

    def build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data.
            self.train_inputs = tf.placeholder(tf.int32, shape=[1,None])
            self.train_labels = tf.placeholder(tf.int32, shape=[1, 1])

            emb = tf.random_uniform([self.vocabulary_size_emb, self.embedding_size], -1.0, 1.0)

            # Look up embeddings for inputs.
            self.embeddings = tf.Variable(initial_value=emb, trainable=True)
            embed=tf.reduce_sum(tf.nn.embedding_lookup(self.embeddings, self.train_inputs),axis=1)

            nce_weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                    stddev=1.0 / math.sqrt(self.embedding_size)))
            nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]),dtype=tf.float32)

            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                 biases=nce_biases,
                                                 inputs=embed,
                                                 labels=self.train_labels,
                                                 num_sampled=self.neg_samples,
                                                 num_classes=self.vocabulary_size))

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            lr = tf.train.exponential_decay(0.01, global_step=self.global_step, decay_steps=int(0.05*self.iter),
                                            decay_rate=0.95)
            self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, global_step=self.global_step)

            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / norm



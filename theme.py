import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, vocab_size, num_nodes, batch_size, embed_size, walk_length, sentence_len):
        self.walk_length = walk_length
        self.embed_size = embed_size
        self.num_nodes = num_nodes
        with tf.name_scope('inputs') as scope:
            self.Text_a = tf.placeholder(tf.int32, [batch_size, sentence_len], name='Ta')
            self.Theme_a = tf.placeholder(tf.int32, [batch_size, walk_length, sentence_len], name='Theme_a')
            self.Text_b = tf.placeholder(tf.int32, [batch_size, sentence_len], name='Tb')
            self.Theme_b = tf.placeholder(tf.int32, [batch_size, walk_length, sentence_len], name='Theme_b')
            self.Text_neg = tf.placeholder(tf.int32, [batch_size, sentence_len], name='Tneg')
            self.Theme_neg = tf.placeholder(tf.int32, [batch_size, walk_length, sentence_len], name='Theme_neg')
        with tf.name_scope('embedding') as scope:
            self.text_embed = tf.Variable(tf.truncated_normal([int(vocab_size), int(embed_size)], stddev=0.3))
            self.attention = tf.Variable(tf.diag(
                np.random.uniform(-1. / np.sqrt(self.num_nodes), 1. / np.sqrt(self.num_nodes), (self.embed_size,))))
        with tf.name_scope('lookup') as scope:
            self.TA = tf.nn.embedding_lookup(self.text_embed, self.Text_a)  # (39, 100, 50)
            self.T_A = tf.expand_dims(self.TA, -1)  # (39,100,50,1)
            self.TB = tf.nn.embedding_lookup(self.text_embed, self.Text_b)
            self.T_B = tf.expand_dims(self.TB, -1)
            self.TNEG = tf.nn.embedding_lookup(self.text_embed, self.Text_neg)
            self.T_Neg = tf.expand_dims(self.TNEG, -1)
            # batch_size=39 每个theme包含2个节点 每个节点100个词 每个词50维
            self.ThemeA = tf.nn.embedding_lookup(self.text_embed, self.Theme_a)  # (39, 2, 100, 50)
            self.ThemeB = tf.nn.embedding_lookup(self.text_embed, self.Theme_b)
            self.ThemeNeg = tf.nn.embedding_lookup(self.text_embed, self.Theme_neg)

        self.A, self.B, self.Neg = self.conv()
        self.loss = self.loss_func()

    def conv(self):
        W = tf.Variable(tf.truncated_normal([2, int(self.embed_size), 1, 50], stddev=0.3))
        themeA = tf.reduce_mean(self.ThemeA, axis=1)  # (39,100,50)
        themeB = tf.reduce_mean(self.ThemeB, axis=1)
        themeNeg = tf.reduce_mean(self.ThemeNeg, axis=1)

        convA = tf.nn.conv2d(self.T_A, W, strides=[1, 1, 1, 1], padding='VALID')  # (39, 99, 1, 50)
        convB = tf.nn.conv2d(self.T_B, W, strides=[1, 1, 1, 1], padding='VALID')
        convNeg = tf.nn.conv2d(self.T_Neg, W, strides=[1, 1, 1, 1], padding='VALID')

        convThemeA = tf.nn.conv2d(tf.expand_dims(themeA, -1), W, strides=[1, 1, 1, 1], padding='VALID')  # (39,99,1,50)
        convThemeB = tf.nn.conv2d(tf.expand_dims(themeB, -1), W, strides=[1, 1, 1, 1], padding='VALID')
        convThemeNeg = tf.nn.conv2d(tf.expand_dims(themeNeg, -1), W, strides=[1, 1, 1, 1], padding='VALID')

        hA = tf.tanh(tf.squeeze(convA))  # (39,99,50)
        hB = tf.tanh(tf.squeeze(convB))
        hNeg = tf.tanh(tf.squeeze(convNeg))

        hThemeA = tf.tanh(tf.squeeze(convThemeA))
        hThemeB = tf.tanh(tf.squeeze(convThemeB))
        hThemeNeg = tf.tanh(tf.squeeze(convThemeNeg))

        pool_A = tf.reduce_mean(hA, 1)  # (39,50)
        pool_B = tf.reduce_mean(hB, 1)
        pool_Neg = tf.reduce_mean(hNeg, 1)

        pool_hThemeA = tf.reduce_mean(hThemeA, 1)  # (39,50)
        pool_hThemeB = tf.reduce_mean(hThemeB, 1)
        pool_hThemeNeg = tf.reduce_mean(hThemeNeg, 1)

        attA = tf.nn.softmax(-(pool_A - pool_hThemeA) ** 2, axis=0)  # (39,50)
        attB = tf.nn.softmax(-(pool_B - pool_hThemeB) ** 2, axis=0)  # (39,50)
        attNeg = tf.nn.softmax(-(pool_Neg - pool_hThemeNeg) ** 2, axis=0)  # (39,50)

        A = pool_A + attA * pool_hThemeA
        B = pool_B + attB * pool_hThemeB
        Neg = pool_Neg + attNeg * pool_hThemeNeg

        return A, B, Neg

    def loss_func(self):
        loss1 = tf.log(tf.sigmoid(tf.reduce_mean(tf.multiply(self.A, self.B), 1)) + 0.001)
        loss2 = tf.log(tf.sigmoid(-tf.reduce_mean(tf.multiply(self.A, self.Neg), 1)) + 0.001)
        loss3 = tf.log(tf.sigmoid(-tf.reduce_mean(tf.multiply(self.B, self.Neg), 1)) + 0.001)
        loss = -tf.reduce_mean(loss1+loss2+loss3)
        return loss

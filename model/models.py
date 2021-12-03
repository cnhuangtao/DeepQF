# -*- coding: utf-8 -*-

#author: huangtao15 
#date: 20201113

import tensorflow as tf
from model_parts import *
from rnn import dynamic_rnn

class Model(object):
    def __init__(self, params):
        #参数列表
        self.embedding_dim = params['embedding_dim']
        self.dropout_rate = params['dropout_rate']
        self.lr = params['lr']
        self.l2 = params['l2']
        self.num_heads = params['num_heads']
        self.model_type = params['model_type']
        
        self.seq_len = params['seq_len']
        self.cateid_size = params['cateid_size']
        self.localid_size = params['localid_size']
        self.onehot_num_list = params['onehot_num_list']
        self.con_fea_len = params['con_fea_len']
        self.pos_len = params["pos_len"]
        self.pra_len = params["pra_len"]
        self.batch_size = params["batch_size"]
        self.pre_weight = params["pre_weight"]
        self.show_weight = params["show_weight"]
        self.click_weight = params["click_weight"]
        self.gcn_len = params["gcn_len"]
        self.label_expend = params["label_expend"]
        self.use_att = params["use_att"]
        self.use_pra = params["use_pra"]
        self.use_gcn = params["use_gcn"]
        self.use_pos = params["use_pos"]
        self.use_cnn = params["use_cnn"]
        
        #数据placeholder
        self.seqno = tf.placeholder(tf.string, [None, ], name='seqno')#batch_size 
        self.click = tf.placeholder(tf.string, [None, ], name='click')#batch_size 
        self.is_train = tf.placeholder_with_default(False, (), 'is_train')
        self.label_range = tf.placeholder(tf.int32, [None, 300], name='label_range')#batch_size * 300
        self.click_label = tf.placeholder(tf.float32, [None, 300], name='click_label')#batch_size * 300
        self.label_len = tf.placeholder(tf.int32, [None, ], name='label_len')#batch_size
        self.show_label = tf.placeholder(tf.float32, [None, 300], name='show_label')#batch_size *  300
        self.pre_label = tf.placeholder(tf.float32, [None, 300], name='pre_label')#batch_size * 300
        self.user_localid = tf.placeholder(tf.int32, [None, ], name='user_localid')#batch_size 
        self.user_cateid = tf.placeholder(tf.int32, [None, ], name='user_cateid')#batch_size 
        self.con_feature = tf.placeholder(tf.float32, [None, self.con_fea_len], name='con_feature')#batch_size * con_fea_len
        self.cate_feature = tf.placeholder(tf.int32, [None, len(self.onehot_num_list)], name='category_feature')#batch_size*onehot_num
        self.pra_localid = tf.placeholder(tf.int32, [None, self.pra_len], name='pra_localid')#batch_size *  100
        self.pra_cateid = tf.placeholder(tf.int32, [None, self.pra_len], name='pra_cateid')#batch_size *  100
        self.max_localid = tf.placeholder(tf.int32, [None, self.pos_len], name='max_localid')#batch_size *  120
        self.max_cateid = tf.placeholder(tf.int32, [None, self.pos_len], name='max_cateid')#batch_size *  120
        self.pos_click = tf.placeholder(tf.float32, [None, self.pos_len], name='pos_click')#batch_size *  120
        self.localid_seq = tf.placeholder(tf.int32, [None, self.seq_len], name='localid_seq')#batch_size *  30
        self.cateid_seq = tf.placeholder(tf.int32, [None, self.seq_len], name='cateid_seq')#batch_size *  30
        self.mask = tf.placeholder(tf.float32, [None, self.seq_len], name='mask')#batch_size *  30
        self.gcn_cateid = tf.placeholder(tf.int32, [None, self.gcn_len], name='gcn_cateid')#batch_size *  30
        self.gcn_cateid_d = tf.placeholder(tf.float32, [None, self.gcn_len], name='gcn_cateid_d')#batch_size *  30
        self.gcn_cateid_num = tf.placeholder(tf.float32, [None, self.gcn_len], name='gcn_cateid_num')#batch_size *  30
        #构建模型
        self.debug = tf.constant([0],dtype = tf.int32)#测试输出用
        self.forward()
        
    def forward(self):
        '''前馈网络
        '''
        with tf.name_scope('Embedding_layer'):
            #localid embedding
            #localid_var = tf.get_variable("localid_var", [self.localid_size, self.embedding_dim])
            #user_localid_embedding = tf.nn.embedding_lookup(localid_var, self.user_localid)#batch_size * embedding_dim
            #pra_localid_embedding = tf.nn.embedding_lookup(localid_var, self.pra_localid)#batch_size*100*embedding_dim
            #max_localid_embedding = tf.nn.embedding_lookup(localid_var, self.max_localid)#batch_size*120*embedding_dim
            #localid_seq_embedding = tf.nn.embedding_lookup(localid_var, self.localid_seq)#batch_size*30*embedding_dim
            
            #cateid embedding
            cateid_var = tf.get_variable("cateid_var", [self.cateid_size, self.embedding_dim])
            user_cateid_embedding = tf.nn.embedding_lookup(cateid_var, self.user_cateid)#batch_size  * embedding_dim
            pra_cateid_embedding = tf.nn.embedding_lookup(cateid_var, self.pra_cateid)#batch_size*100*embedding_dim
            max_cateid_embedding = tf.nn.embedding_lookup(cateid_var, self.max_cateid)#batch_size*120*embedding_dim
            cateid_seq_embedding = tf.nn.embedding_lookup(cateid_var, self.cateid_seq)#batch_size*30*embedding_dim
            gcn_cateid_embedding = tf.nn.embedding_lookup(cateid_var, self.gcn_cateid)#batch_size*30*embedding_dim
            
            #feature embedding
            category_embedding_var = []
            for i in range(len(self.onehot_num_list)):
                if i in [0,7,8,9]:
                    category_embedding_var += [tf.get_variable(name = 'onehot_'+str(i),shape=[self.onehot_num_list[i], self.embedding_dim])]
            category_feature_embedding = []
            for i,d in enumerate([0,7,8,9]):
                per_feature = self.cate_feature[:, d]
                category_feature_embedding += [tf.nn.embedding_lookup(category_embedding_var[i], per_feature)]#[batch_size * 1 * embedding_dim]
            
        with tf.variable_scope('click_seq_attention'):
            seq_mask = tf.reshape(self.mask, [-1,self.seq_len, 1])#batch_size * seq_len  *1
            seq_sum_mask = cateid_seq_embedding * seq_mask##batch_size * seq_len  *embedding_dim
            seq_real_len = tf.reduce_sum(self.mask,1, keepdims = True)#batch_size *1
            seq_sum = tf.reduce_sum(seq_sum_mask, 1) / (seq_real_len) #batch_size  * embedding_dim
            #attention
            src_mask_click = tf.math.equal(self.mask, 0)#batch_size * seq_len #binary
            seqs = tf.layers.dropout(cateid_seq_embedding, self.dropout_rate, training=self.is_train)
            seqs = multihead_attention(seqs, seqs, seqs, src_mask_click, num_heads=8, dropout_rate=self.dropout_rate, training=self.is_train)
            seqs = ff(seqs, [128, self.embedding_dim])#batch_size  * seq_len * embedding_dim) 
            seqs = seqs * tf.expand_dims(self.mask, -1)#batch_size  * seq_len * embedding_dim
            seq_emb = tf.reduce_sum(seqs, axis=1) / (seq_real_len)#batch_size  * embedding_dim
        
        with tf.variable_scope('pra_process'):
            pra_feature = tf.reshape(pra_cateid_embedding, [-1, self.pra_len * self.embedding_dim])

        with tf.variable_scope('gcn_process'):
            channel_d = tf.reduce_sum(tf.cast(self.gcn_cateid_num,tf.float32),1, keepdims = True)#batch_size *1
            item_d = self.gcn_cateid_d#batch_size * gcn_len
            item_e = self.gcn_cateid_num#batch_size * gcn_len
            item_x = gcn_cateid_embedding#batch_size * gcn_len * embedding_dim
            gcn_weight = tf.reshape(tf.rsqrt(item_d + 0.0000001) * tf.rsqrt(channel_d + 0.0000001) * item_e, [-1,self.gcn_len,1])#batch_size*gcn_len*1
            gcn_out = tf.layers.dense(gcn_weight * item_x, self.embedding_dim, activation=None)#batch*gcn_len*embedding_dim
            gcn_out = tf.reduce_sum(gcn_out, 1)#batch_size * embedding_dim
            gcn_out = tf.layers.dense(gcn_out, self.embedding_dim, activation="tanh")#batch*embedding_dim
            
        with tf.variable_scope('conv_process'):
            pos_embedding = max_cateid_embedding#batch*pos_len*embedding_dim
            pos_mask = tf.cast(tf.not_equal(self.max_cateid,0),tf.float32)#batch_size * pos_len
            pos_embedding = pos_embedding * tf.reshape(pos_mask, [-1,self.pos_len, 1])#batch*pos_len*embedding_dim
            pos_real_len = tf.reduce_sum(pos_mask,1, keepdims = True)#batch_size *1
            pos_sum = tf.reduce_sum(pos_embedding, 1) / (pos_real_len+0.0000001) #batch_size  * embedding_dim
            #cnn
            conv = tf.layers.conv1d(pos_embedding, self.embedding_dim, 3)#batch*(pos_len-2)*self.embedding_dim
            conv_out = tf.reduce_max(conv, reduction_indices=[1])#batch*self.embedding_dim
            
        #concat feature
        features = [user_cateid_embedding] + category_feature_embedding + [seq_sum]
        if self.use_att:
            features += [seq_emb]
        if self.use_pra:
            features += [pra_feature]
        if self.use_gcn:
            features += [gcn_out]
        if self.use_pos:
            features += [self.pos_click]
        if self.use_cnn:
            features += [pos_sum, conv_out]
        #————————————————C-Net————————————————————
        with tf.variable_scope("a-Net"):
            inp = tf.concat(features, -1)
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)#
            dnn1 = tf.layers.dense(bn1, 1024, activation=None, name='f1', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2))
            dnn1 = tf.nn.tanh(dnn1, 'relu1')
            dnn1 = tf.layers.dropout(dnn1, self.dropout_rate, training=self.is_train)
            dnn2 = tf.layers.dense(dnn1, 256, activation=None, name='f2', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2))
            dnn2 = tf.layers.batch_normalization(inputs=dnn2, name='bn2', training=self.is_train)
            dnn2 = tf.nn.tanh(dnn2, 'relu2')
            dnn2 = tf.layers.dropout(dnn2, self.dropout_rate, training=self.is_train)
            dnn3 = tf.layers.dense(dnn2, 1, activation=None, name='f3')
            #self.a = tf.nn.relu(dnn3, 'relu3') + 0.0000001
            self.a = tf.square(dnn3) + 0.0000001
        with tf.variable_scope("b-Net"):
            inp = tf.concat(features, -1)
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)#
            dnn1 = tf.layers.dense(bn1, 1024, activation=None, name='f1', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2))
            dnn1 = tf.nn.tanh(dnn1, 'relu1')
            dnn1 = tf.layers.dropout(dnn1, self.dropout_rate, training=self.is_train)
            dnn2 = tf.layers.dense(dnn1, 256, activation=None, name='f2', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2))
            dnn2 = tf.layers.batch_normalization(inputs=dnn2, name='bn2', training=self.is_train)
            dnn2 = tf.nn.tanh(dnn2, 'relu2')
            dnn2 = tf.layers.dropout(dnn2, self.dropout_rate, training=self.is_train)
            dnn3 = tf.layers.dense(dnn2, 1, activation=None, name='f3')
            #self.b = tf.nn.relu(dnn3, 'relu3') + 0.0001
            self.b = tf.square(dnn3) + 0.0000001
        with tf.name_scope('Label'):    
            label_max_len = 200#label的最长长度
            label_range = tf.tile(tf.reshape(tf.range(label_max_len), (1,-1)),[label_max_len,1])#150*150
            label_mask = tf.linalg.band_part(tf.ones((label_max_len,label_max_len)),-1,0)#150*150
            label_martrix = tf.cast(label_range, tf.float32) * label_mask#150*150
            label_len = tf.cast(tf.round((self.label_len - 1)/2 * self.label_expend),tf.int32)#真实的长度
            label_range = tf.nn.embedding_lookup(label_martrix, label_len)#batch_size * 150
            click_label = self.click_label[:,:label_max_len]
            show_label = self.show_label[:,:label_max_len]
            #pre_label = self.pre_label[:,:label_max_len]
            
        with tf.name_scope('Metrics'):
            self.info = tf.as_string(self.cate_feature[:,0])+"#"+tf.as_string(self.user_cateid)
            #target = self.click_weight*click_label+self.show_weight*show_label+self.pre_weight*pre_label#定义目标
            target = self.click_weight*click_label+self.show_weight*show_label#定义目标
            mse_loss = tf.reduce_mean(tf.reduce_mean(tf.square(self.a * tf.log(label_range * self.b + 1) - target), 1))
            l2_loss = tf.losses.get_regularization_loss()
            self.loss = l2_loss + mse_loss
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.debug = tf.concat([tf.reduce_sum(pos_embedding, 1),pos_sum], -1)
        
    def evaluate(self, sess, inps,  is_train = False):
        feed_dict = {
            self.seqno: inps[0],
            self.click: inps[1],
            self.label_len: inps[2],
            self.label_range: inps[3],
            self.click_label: inps[4],
            self.show_label: inps[5],
            self.pre_label: inps[6],
            self.user_localid: inps[7],
            self.user_cateid: inps[8],
            self.con_feature: inps[9],
            self.cate_feature: inps[10],
            self.pra_localid: inps[11], 
            self.pra_cateid: inps[12],
            self.max_localid: inps[13],
            self.max_cateid: inps[14],
            self.pos_click: inps[15],
            self.localid_seq: inps[16],
            self.cateid_seq: inps[17],
            self.mask: inps[18],
            self.gcn_cateid: inps[19],
            self.gcn_cateid_d: inps[20],
            self.gcn_cateid_num: inps[21],
            self.is_train:is_train
        }
        seqno,click,info,a,b,loss = sess.run([self.seqno,self.click,self.info,self.a, self.b, self.loss], feed_dict=feed_dict)
        return seqno,click,info,a,b,loss

    def train(self, sess, inps, is_train = True):
        feed_dict = {
            self.seqno: inps[0],
            self.click: inps[1],
            self.label_len: inps[2],
            self.label_range: inps[3],
            self.click_label: inps[4],
            self.show_label: inps[5],
            self.pre_label: inps[6],
            self.user_localid: inps[7],
            self.user_cateid: inps[8],
            self.con_feature: inps[9],
            self.cate_feature: inps[10],
            self.pra_localid: inps[11], 
            self.pra_cateid: inps[12],
            self.max_localid: inps[13],
            self.max_cateid: inps[14],
            self.pos_click: inps[15],
            self.localid_seq: inps[16],
            self.cateid_seq: inps[17],
            self.mask: inps[18],
            self.gcn_cateid: inps[19],
            self.gcn_cateid_d: inps[20],
            self.gcn_cateid_num: inps[21],
            self.is_train:is_train
        }
        loss, _, _,debug = sess.run([self.loss, self.optimizer, self.update_ops, self.debug], feed_dict=feed_dict)
        return loss,debug

    def save(self, sess, path, steps):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path, global_step=steps)
                
        
        
        
        
        
        
        
        
        
        
        
        

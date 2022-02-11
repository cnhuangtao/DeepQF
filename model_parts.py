import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
import numpy as np

def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_" + scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc


def calc_gauc(raw_arr_dict):
    gauc = 0.0
    cnt = 0
    imei_count = 0
    for imei, raw_arr in raw_arr_dict.items():
        if 1. not in np.array(raw_arr)[:, 1] or 0. not in np.array(raw_arr)[:, 1]:
            print(imei)
            continue
        auc = calc_auc(raw_arr)
        gauc += auc * len(raw_arr)
        cnt += len(raw_arr)
        imei_count += 1
    gauc = gauc / cnt
    print("Impressions: {0}, Imeis: {1}, All Click and Zero Click Imeis: {2} ".format(cnt, len(raw_arr_dict.keys()), len(raw_arr_dict.keys())-imei_count))
    # tf.logging.info("Impressions: %d, Imeis: %d, All Click and Zero Click Imeis: %d ", cnt, len(raw_arr_dict.keys()), len(raw_arr_dict.keys())-imei_count)
    return gauc


def attention(query, facts, attention_size, mask, stag='null', mode='LIST', softmax_stag=1, time_major=False,
              return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])

    mask = tf.equal(mask, tf.ones_like(mask))
    hidden_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer 3
    input_size = query.get_shape().as_list()[-1]  # AD embedding 3

    # Trainable parameters
    w1 = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))  # 3*3
    w2 = tf.Variable(tf.random_normal([input_size, attention_size], stddev=0.1))  # 3*attention_size
    b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    v = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # query: [batch_size,embedding] facts:[batch_size,his,embedding] batch_size=2,his=3,embedding=3
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `tmp` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        # (batch_size,his,embedding)*(embedding,attention_size)=(batch_size,his,attention_size)
        tmp1 = tf.tensordot(facts, w1, axes=1)
        # (batch_size,embedding)*(embedding,attention_size)=(batch_size,attention_size)
        tmp2 = tf.tensordot(query, w2, axes=1)
        # (batch_size,1,attention_size)
        tmp2 = tf.reshape(tmp2, [-1, 1, tf.shape(tmp2)[-1]])
        # (batch_size,his,attention_size) + (batch_size,1,attention_size)=(batch_size,his,attention_size)
        tmp = tf.tanh((tmp1 + tmp2) + b)

    # For each of the timestamps its vector of size A from `tmp` is reduced with `v` vector
    # (batch_size,his,attention_size)*(attention_size)
    v_dot_tmp = tf.tensordot(tmp, v, axes=1, name='v_dot_tmp')  # (B,T) shape
    key_masks = mask  # [B, 1, T]
    # key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(v_dot_tmp) * (-2 ** 32 + 1)
    v_dot_tmp = tf.where(key_masks, v_dot_tmp, paddings)  # [B, 1, T]
    alphas = tf.nn.softmax(v_dot_tmp, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    # output = tf.reduce_sum(facts * tf.expand_dims(alphas, -1), 1)
    output = facts * tf.expand_dims(alphas, -1)
    output = tf.reshape(output, tf.shape(facts))
    # output = output / (facts.get_shape().as_list()[-1] ** 0.5)
    if not return_alphas:
        return output
    else:
        return output, alphas


def din_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                  return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print("querry_size mismatch")
        query = tf.concat(values=[
            query,
            query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # [[1,1,1,0,0],[1,1,1,1,1]]-> [[true,true,true,false,false],[true,true,true,true,true]]
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    # query: [batch_size,embedding] facts:[batch_size,his,embedding] batch_size=2,his=3,embedding=3
    queries = tf.tile(query, [1, tf.shape(facts)[1]])  # [[1,2,4],[3,4,6]]->[[1,2,4,1,2,4,1,2,4],[3,4,6,3,4,6,3,4,6]]
    queries = tf.reshape(queries, tf.shape(
        facts))  # [[1,2,4,1,2,4,1,2,4],[3,4,6,3,4,6,3,4,6]]->[[[1,2,4][1,2,4][1,2,4]],[[3,4,6][3,4,6][3,4,6]]]
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 40, activation=tf.nn.sigmoid, name='f1_att' + stag)  # [2,3,40]
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 20, activation=tf.nn.sigmoid, name='f2_att' + stag)  # [2,3,20]
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)  # [2,3,1]
    d_layer_3_all = tf.reshape(d_layer_3_all,
                               [-1, 1, tf.shape(facts)[1]])  # [2,1,3]
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    # 作用是将scores中对应key_masks中true的位置的元素值不变，其余元素替换成paddings中对应位置的元素值
    # [[true,true,true,false,false],[true,true,true,true,true]]
    #
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T] [[[0.3][0.2][0.5]],[[0.4][0.5][0.1]]]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output

def din(querys, keys, mask):
    """
        queries:     [Batchsize, 1, embedding_size]
        keys:        [Batchsize, max_seq_len, embedding_size]
        mask:     [Batchsize, max_seq_len]
    """

    keys_length = tf.shape(keys)[1] #
    embedding_size = querys.get_shape().as_list()[-1]
    keys = tf.reshape(keys, shape=[-1, keys_length, embedding_size])#batch * n * embedding
    querys = tf.reshape(tf.tile(querys, [1, keys_length, 1]), shape=[-1, keys_length, embedding_size])#batch * n * embedding#在第二维度上复制n

    net = tf.concat([keys, keys - querys, querys, keys*querys], axis=-1)#线性组合的concat
    for units in [32,16]:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)        #batch * n * 1
    outputs = tf.reshape(att_wgt, shape=[-1, 1, keys_length], name="weight")  #batch * 1 * n
    
    scores = outputs #batch * 1 * n
    #key_masks = tf.expand_dims(tf.cast(keys_id > 0, tf.bool), axis=1)  # shape(batch_size, 1, max_seq_len) we add 0 as padding
    # tf.not_equal(keys_id, '0')  如果改成str
    key_masks = tf.expand_dims(mask, axis=1)#batch * 1 * n
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)#满足mask条的Ture位置返回score，否则返回很小的数-2 ** 32 + 1
    scores = scores / (embedding_size ** 0.5)       # scale
    scores = tf.nn.softmax(scores)
    outputs = tf.matmul(scores, keys)    #(batch_size, 1, embedding_size)
    outputs = tf.reduce_sum(outputs, 1, name="attention_embedding")   #(batch_size, embedding_size)
    
    return outputs


def multihead_attention(queries, keys, values, key_masks,
                        num_heads=3,
                        dropout_rate=0.1,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    d_model = queries.get_shape().as_list()[-1]  # batchsize*his*d
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = ln(outputs)

    return outputs


def scaled_dot_product_attention(Q, K, V, key_masks,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, key_masks=key_masks, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # # query masking
        # outputs = mask(outputs, Q, K, type="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs


def mask(inputs, key_masks=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"
    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],
       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]) # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
    # elif type in ("q", "query", "queries"):
    #     # Generate masks
    #     masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
    #     masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
    #     masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)
    #
    #     # Apply masks to inputs
    #     outputs = inputs*masks
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs


def ln(inputs, epsilon=1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    # V = inputs.get_shape().as_list()[-1]  # number of channels
    V = 2
    return ((1 - epsilon) * inputs) + (epsilon / V)


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape




def scaled_dot_product_attention_ht(Q, K, V, mask = None, use_mask = False,dropout_rate=0.2, training=True, scope="scaled_dot_product_attention"):
    '''Attention(Q,K,V)=softmax(Q K^T /√dk ) V
        :param Q: 查询，三维张量，[batch, n_q, d]
        :param K: keys值，三维张量，[batch, n_k, d]
        :param V: values值，三维张量，[batch, n_v, d]
        return [batch, n_q, d], [n_q, d]
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]#d_model/h
        n_q = Q.get_shape().as_list()[1]
        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (h*batch, n_q, n_k)
        # scale 缩放
        outputs /= d_k ** 0.5
        #mask
        if use_mask:
            mask = tf.to_float(mask)#转为浮点数，True的位置会被mask掉# (h*batch, n_k)
            mask = tf.expand_dims(mask, 1) # (h*batch, 1, n_k)
            mask = tf.tile(mask, [1, n_q, 1])# (h*batch, n_q, n_k)
            paddings =-2 ** 32 + 1
            outputs = outputs + mask * paddings
        # softmax
        attention = tf.nn.softmax(outputs)# (h*batch, n_q, n_k)
        
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)
    return outputs, attention

def multihead_attention_ht(queries, keys, values, mask = None, num_heads=8, d_model = 512, dropout_rate=0.2, training=True, scope="multihead_attention", use_mask = False):
    '''
        :param queries: 三维张量[batch, n_q, d_q]
        :param d_model: 模型的运算维度
        :param keys: 三维张量[batch, n_k, d_k]
        :param values: 三维张量[batch, n_k, d_v]
        :param num_heads: heads数
        :param dropout_rate:
        :param training: 控制dropout机制
        :mask: [batch, n_k]
        :return: 三维张量(batch, n_q, d_model)
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 将QKV变换为d_model
        Q = tf.layers.dense(queries, d_model, use_bias=False) # (batch, n_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=False) # (batch, n_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=False) # (batch, n_k, d_model)
        # 把d_model分为num_heads份数
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*batch, n_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*batch, n_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*batch, n_k, d_model/h)
        if use_mask:
            mask = tf.tile(mask, [num_heads, 1])# (h*batch, n_k,)
        # Attention
        outputs, _ = scaled_dot_product_attention_ht(Q_, K_, V_, mask = mask, use_mask = use_mask, dropout_rate=dropout_rate, training = training)# (h*batch, T_q, d_model/h)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (batch, n_q, d_model)
        # 残差结构
        outputs += queries
        # Normalize
        outputs = ln(outputs)
        #outputs = tf.layers.batch_normalization(outputs, momentum=0.999, training=training)
    return outputs
    
def feedforward_ht(inputs, d_ffn = 2048, d_model = 512, training=True, dropout_rate=0.8, scope="positionwise_feedforward"):
    '''inputs: A 3d tensor with shape of [batch, n, d_model].
    Returns:[batch, n, d_model]
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, d_ffn, activation=tf.nn.relu)
        # Outer layer
        outputs = tf.layers.dense(outputs, d_model)
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = ln(outputs)
        #outputs = tf.layers.batch_normalization(outputs, momentum=0.999, training=training)
    return outputs

def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                      return_alphas=False, forCnn=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output
class VecAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        # reset and update
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h

# -*- coding: utf-8 -*-

#author: huangtao15 
#date: 20210803

import os
import numpy
from models import *
import random
import sys
import time
import datetime
from tqdm import tqdm
import tensorflow as tf
import glob
import json
import pickle as pkl
import pandas as pd
from functools import wraps
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.saved_model import constants
from tensorflow import contrib
from sklearn.metrics import roc_auc_score
from sklearn import metrics

autograph = contrib.autograph
rootPath = os.getcwd()
sys.path.append(rootPath)
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#----------------------------------设置模型参数----------------------------------------------
tf.app.flags.DEFINE_string('f','' ,'kernel')
FLAGS = tf.app.flags.FLAGS

#路径参数
tf.app.flags.DEFINE_string("model_dir", './model/', "model dir")
tf.app.flags.DEFINE_string("data_dir", './data/', "data dir")
tf.app.flags.DEFINE_string("output_dir", './data/result/', "output_dir")
tf.app.flags.DEFINE_string("export_dir", './export_model_dir/', "export model dir")
tf.app.flags.DEFINE_string("hdfs_dir", 'hdfs://default/home/hdp_teu_search/resultdata/recommend/huangtao15/main_recall/sample', "hdfs dir")
tf.app.flags.DEFINE_string("hdfs_out_path", '/home/hdp_teu_search/resultdata/recommend/huangtao15/main_recall/predict',"hdfs_out_path")

#设定时间为当前的前一天
now_time = datetime.datetime.now()-datetime.timedelta(days=1)
now_time = now_time.strftime("%Y%m%d")
tf.app.flags.DEFINE_string("date", now_time, "train_part")
#数据参数
tf.app.flags.DEFINE_integer("seq_len", 30, "seq_len")
tf.app.flags.DEFINE_integer("gcn_len", 30, "gcn_len")
tf.app.flags.DEFINE_integer("pos_len", 120, "pos_len")
tf.app.flags.DEFINE_integer("pra_len", 100, "pra_len")
tf.app.flags.DEFINE_integer("date_span", 7, "date_span")
tf.app.flags.DEFINE_integer("input_file_start", 0, "input_file_start")
tf.app.flags.DEFINE_integer("input_file_end", 100, "input_file_end")
tf.app.flags.DEFINE_integer("cateid_size",14000, "cateid_size")#cateid数量
tf.app.flags.DEFINE_integer("localid_size",120000, "localid_size")#localid数量
tf.app.flags.DEFINE_string("onehot_num_list", "[2100,2,2,2,2,2,5,24,8,2,2861,2861]", "onehot_num list")#phone 2861暂时没加
tf.app.flags.DEFINE_integer("con_fea_len", 9, "onehot_numcon_fea_lenlist")

#模型参数
tf.app.flags.DEFINE_integer("embedding_dim", 16, "embedding_dim")
tf.app.flags.DEFINE_integer("num_heads", 8, "num_heads")
tf.app.flags.DEFINE_float("lr", 0.0001, "dnn learning rate")#学习率
tf.app.flags.DEFINE_float("dropout_rate", 0.0, "dropout_rate")#droup out
tf.app.flags.DEFINE_float("l2", 0.003, "l2")#l2_reg
tf.app.flags.DEFINE_float("pre_weight", 0, "pre_weight")#label中先验的权值
tf.app.flags.DEFINE_float("show_weight", 0.1, "show_weight")#label中曝光的权值
tf.app.flags.DEFINE_float("click_weight", 1, "click_weight")#label中点击的权值
tf.app.flags.DEFINE_float("label_expend", 1.1, "label_expend")#label扩展倍数

#训练参数
tf.app.flags.DEFINE_integer("train_epochs",1, "train epochs")#训练迭代次数
tf.app.flags.DEFINE_integer("log_steps", 10, "log_steps")
tf.app.flags.DEFINE_integer("cpu_num", 10, "cpu_num")
tf.app.flags.DEFINE_integer("batch_size", 1024, "batch_size_train")
tf.app.flags.DEFINE_integer("eval_per_num", 100, "eval_per_num")#调用test频率
tf.app.flags.DEFINE_integer("save_iter", 50, "save_iter")#保存模型step
tf.app.flags.DEFINE_integer("is_part_test",1, "is_part_test")#是否只是用部分测试集
tf.app.flags.DEFINE_integer("batch_size_total_test", 50, "batch_size_total_test")#测试step上限
tf.app.flags.DEFINE_integer("local_train",1, "local_train")
tf.app.flags.DEFINE_integer("file_zise", 20, "file_zise")
tf.app.flags.DEFINE_integer("train_task_no", 0, "train_task_no")

#消融实验
tf.app.flags.DEFINE_integer("use_att", 1, "use_att")
tf.app.flags.DEFINE_integer("use_pra", 1, "use_pra")
tf.app.flags.DEFINE_integer("use_gcn", 1, "use_gcn")
tf.app.flags.DEFINE_integer("use_pos", 1, "use_pos")
tf.app.flags.DEFINE_integer("use_cnn", 1, "use_cnn")

tf.app.flags.DEFINE_string("model_type", 'debug', "model_type")#lr,mlp,din,dien,click,2-channel,3-channel,hian
tf.app.flags.DEFINE_integer("save_result", 1, "save_result")

#----------读取数据------------------
def tf_record_input_fn(csv_path, is_train = True):
    '''
    :csv_path:数据路径
    '''
    print('#' * 100)
    print('Loading data: ', csv_path[0])
    def _parse_dl_sample_fn(record):
        features = {
            "seqno": tf.FixedLenFeature([], tf.string),
            "label_len": tf.FixedLenFeature([], tf.int64),
            "click_label": tf.FixedLenFeature([300], tf.float32),
            "user_localid": tf.FixedLenFeature([], tf.int64),
            "user_cateid": tf.FixedLenFeature([], tf.int64),
            "con_feature": tf.FixedLenFeature([9], tf.float32),
            "cate_feature": tf.FixedLenFeature([12], tf.int64),
            "pra_localid": tf.FixedLenFeature([100], tf.int64),
            "pra_cateid": tf.FixedLenFeature([100], tf.int64),
            "max_localid": tf.FixedLenFeature([120], tf.int64),
            "max_cateid": tf.FixedLenFeature([120], tf.int64),
            "pos_click": tf.FixedLenFeature([120], tf.float32),
            "localid_seq": tf.FixedLenFeature([30], tf.int64),
            "cateid_seq": tf.FixedLenFeature([30], tf.int64),
            "mask": tf.FixedLenFeature([30], tf.float32),
            "click": tf.FixedLenFeature([], tf.string),
            "label_range": tf.FixedLenFeature([300], tf.int64),
            "infor_list": tf.FixedLenFeature([300], tf.float32),
            "gcn_cateid": tf.FixedLenFeature([30], tf.int64),
            "gcn_cateid_d": tf.FixedLenFeature([30], tf.float32),
            "gcn_cateid_num": tf.FixedLenFeature([30], tf.float32),
            "show_label": tf.FixedLenFeature([300], tf.float32)
        }
        parsed = tf.parse_single_example(record, features)
        seqno = parsed['seqno']
        click = parsed['click']
        label_len = parsed['label_len']
        click_label = parsed['click_label']
        show_label = parsed['show_label']
        user_localid = parsed['user_localid']
        user_cateid = parsed['user_cateid']
        con_feature = parsed['con_feature']
        cate_feature = parsed['cate_feature']
        pra_localid = parsed['pra_localid']
        pra_cateid = parsed['pra_cateid']
        max_localid = parsed['max_localid']
        max_cateid = parsed['max_cateid']
        pos_click = parsed['pos_click']
        localid_seq = parsed['localid_seq']
        cateid_seq = parsed['cateid_seq']
        mask = parsed['mask']
        label_range = parsed['label_range']
        pre_label = parsed['infor_list']
        gcn_cateid = parsed['gcn_cateid']
        gcn_cateid_d = parsed['gcn_cateid_d']
        gcn_cateid_num = parsed['gcn_cateid_num']
        return seqno,click,label_len,label_range,click_label,show_label,pre_label,user_localid,user_cateid,\
                con_feature,cate_feature,pra_localid,pra_cateid,max_localid,max_cateid,pos_click,\
                localid_seq,cateid_seq,mask,gcn_cateid,gcn_cateid_d,gcn_cateid_num
    cpu_num = FLAGS.cpu_num
    batch_size = FLAGS.batch_size
    files = tf.data.Dataset.list_files(csv_path)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=cpu_num, sloppy=True))
    dataset = dataset.map(_parse_dl_sample_fn, num_parallel_calls=cpu_num).prefetch(tf.contrib.data.AUTOTUNE)
    if is_train:
        dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)
    #dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return dataset

#-------------------保存模型-----------------
def export_model(session, model):
    print('----->export_model...')
    sys.stdout.flush()
    #输入与输出的placeholder
    inputs = {
        "user_localid": utils.build_tensor_info(model.user_localid),
        "user_cateid": utils.build_tensor_info(model.user_cateid),
        "con_feature": utils.build_tensor_info(model.con_feature),
        "cate_feature": utils.build_tensor_info(model.cate_feature),
        "pra_localid": utils.build_tensor_info(model.pra_localid),
        "pra_cateid": utils.build_tensor_info(model.pra_cateid),
        "max_localid": utils.build_tensor_info(model.max_localid),
        "max_cateid":utils.build_tensor_info(model.max_cateid),
        "pos_click": utils.build_tensor_info(model.pos_click),
        "localid_seq": utils.build_tensor_info(model.localid_seq),
        "cateid_seq": utils.build_tensor_info(model.cateid_seq),
        "mask": utils.build_tensor_info(model.mask),
        "gcn_cateid": utils.build_tensor_info(model.gcn_cateid),
        "gcn_cateid_d": utils.build_tensor_info(model.gcn_cateid_d),
        "gcn_cateid_num": utils.build_tensor_info(model.gcn_cateid_num)
    }
    a = utils.build_tensor_info(model.a)
    b = utils.build_tensor_info(model.b)
    info = utils.build_tensor_info(model.info)
    #设置model_signature
    model_signature = signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs={"a": a, "b": b, "info": info},
        method_name=signature_constants.PREDICT_METHOD_NAME)
    t = str(int(time.time()))
    #初始化builder路径
    builder = saved_model_builder.SavedModelBuilder(FLAGS.export_dir + t)
    #将所有参数保存到saver
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]#bn层参数
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list)
    # 用saver和model_signature和op构建builder
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    ops._default_graph_stack.get_default()._collections[constants.MAIN_OP_KEY] = []
    builder.add_meta_graph_and_variables(
            session, [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature,
            }, saver=saver,
            legacy_init_op=legacy_init_op
        )
    builder.save()

#--------------测试函数---------------
def test(sess, test_file, model, n, step, is_part_test = FLAGS.is_part_test):
    dataset_test = tf_record_input_fn(test_file, is_train = False)
    iterator_test = dataset_test.make_initializable_iterator()
    next_element_test = iterator_test.get_next()
    sess.run(iterator_test.initializer)
    test_step = 0#验证集的step
    test_loss_sum = 0
    y_loss = 0#记录loss
    file_counter = 0#保存文件分块的大小
    counter = 0#保存文件的batch数
    writer = open(FLAGS.output_dir + "/part-%04d" % file_counter, "w")#保存路径
    print("【writing】:" + FLAGS.output_dir + "/part-%04d" % file_counter)
    try:#-------按batch读取预测数据--------
        while True:
            test_step += 1
            test_input = sess.run(next_element_test)
            seqno_batch,click_batch,channel_batch,aa,bb,loss = model.evaluate(sess, test_input)
            y_loss += loss
            a = aa[:,0].tolist()
            b = bb[:,0].tolist()
            channel_id = channel_batch.tolist()
            seqno = seqno_batch.tolist()
            click = click_batch.tolist()
            if FLAGS.save_result:#保存文件
                for i in range(len(a)):
                    writer.write(str(seqno[i],encoding = "utf-8") +"\t"+str(channel_id[i],encoding = "utf-8")+"\t" + 
                                 str(a[i]) + "\t" + str(b[i])+"\t"+str(click[i],encoding = "utf-8")+"\n")#写
                counter +=1
                if counter % FLAGS.file_zise == 0:#达到文件大小后更换open
                    writer.close()
                    file_counter += 1
                    writer = open(FLAGS.output_dir + "/part-%04d" % file_counter, "w")
                    print("【writing】:" + FLAGS.output_dir + "/part-%04d" % file_counter)
                    if file_counter == 2 and FLAGS.local_train == 1:#本地训练只保存2个batch
                        writer.close()
                        break    
            if test_step % FLAGS.log_steps == 0:
                print("【test】:mean_loss = %.6f\tbatch = %d\tstep = %d\tepochs = %d" \
                              % (y_loss / test_step, FLAGS.batch_size, test_step, n+1))
                print("【test】:sampling->channel_id=%s\ta=%f\tb=%f\t" %(channel_id[0],\
                                                    a[0],b[0]))
                sys.stdout.flush()
            if test_step >= FLAGS.batch_size_total_test and is_part_test == 1:#阶段测试达到上限输出结果
                print('【END】：loss = %.6f' % (y_loss/test_step))
                sys.stdout.flush()
                break
        writer.close()
    except tf.errors.OutOfRangeError:#---------预测数据读取完毕--------  
        print('#' * 110)
        print('【END】：loss = %.6f' % (y_loss/test_step))
        #上传文件
        if FLAGS.save_result:#保存文件
            hdfs_out_path = FLAGS.hdfs_out_path + "/%s" % FLAGS.train_task_no#每个实验号单独保存hdfs
            try:
                os.system('hadoop fs -rmr ' + hdfs_out_path)
            except:
                print('【HDFS OUT】：old data not found!')
                sys.stdout.flush()
            try:    
                os.system('hadoop fs -mkdir ' + hdfs_out_path)
            except:
                print('【HDFS OUT】：mkdir faild!')
                sys.stdout.flush()
            try:
                os.system('hadoop fs -put ./data/result/part* ' + hdfs_out_path)
            except:
                print('【HDFS OUT】：hdfs put error!')
                sys.stdout.flush()
            print('【HDFS OUT】：save precidt ab success!')
            sys.stdout.flush()

def main():
    #-----------------------数据路径-------------------------
    if FLAGS.local_train:
        train_path = glob.glob("%s/sample/p*" % FLAGS.data_dir)
        test_path = glob.glob("%s/sample/p*" % FLAGS.data_dir)
    else:
        #-----------main-----------
        train_path = []
        test_path = []
        for i in range(FLAGS.input_file_start,FLAGS.input_file_end):
            test_path += ["%s/%s/part-r-00%03d" % (FLAGS.hdfs_dir, FLAGS.date, i)]
        for i in range(FLAGS.date_span):#训练数据7天
            for j in range(FLAGS.input_file_start,FLAGS.input_file_end):
                d = datetime.datetime.strptime(FLAGS.date, '%Y%m%d')-datetime.timedelta(days=i + 1)
                d_str = d.strftime("%Y%m%d")
                train_path += ["%s/%s/part-r-00%03d" % (FLAGS.hdfs_dir, d_str, j)]
        random.shuffle(train_path)
        random.shuffle(test_path)
  
    #---------------------模型参数---------------
    model_params = {
        "seq_len": FLAGS.seq_len,
        "cateid_size":FLAGS.cateid_size,
        "localid_size":FLAGS.localid_size,
        "onehot_num_list":eval(FLAGS.onehot_num_list),
        "con_fea_len":FLAGS.con_fea_len,
        "pos_len":FLAGS.pos_len,
        "pra_len":FLAGS.pra_len,
        "gcn_len":FLAGS.gcn_len,
        "--------------":"---------------",
        "embedding_dim": FLAGS.embedding_dim,
        "lr": FLAGS.lr,
        "dropout_rate":FLAGS.dropout_rate,
        "l2":FLAGS.l2,
        "model_type":FLAGS.model_type,
        "num_heads":FLAGS.num_heads,
        "batch_size":FLAGS.batch_size,
        "show_weight":FLAGS.show_weight,
        "pre_weight":FLAGS.pre_weight,
        "click_weight":FLAGS.click_weight,
        "label_expend":FLAGS.label_expend,
        "use_att":FLAGS.use_att,
        "use_pra":FLAGS.use_pra,
        "use_gcn":FLAGS.use_gcn,
        "use_pos":FLAGS.use_pos,
        "use_cnn":FLAGS.use_cnn
    }
    #初始化sess的config
    config = tf.compat.v1.ConfigProto(device_count={"CPU": FLAGS.cpu_num}, #cpu核数
                                      gpu_options=tf.compat.v1.GPUOptions(allow_growth=True),#动态申请显存
                                      intra_op_parallelism_threads=FLAGS.cpu_num, 
                                      log_device_placement=False, #记录设备指派情况
                                      allow_soft_placement=True)#自动选择设备
    #--------------------training-------------------
    with tf.Session(config=config) as sess:
        model = Model(model_params)
        #读取训练数据
        dataset = tf_record_input_fn(train_path, is_train = True)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        #初始化
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        sys.stdout.flush()
        #开始训练
        step = 0
        for n in range(FLAGS.train_epochs):
            sess.run(iterator.initializer)#每次迭代都要初始化
            #train
            loss_sum = 0
            input_time = 0
            model_time = 0
            try:
                 while True:
                    #取batch数据
                    st = time.time()
                    model_input = sess.run(next_element)#读取数据
                    input_t = time.time()-st#计算读取数据用时
                    if input_time == 0:
                        input_time = input_t
                    else:
                        input_time = (input_time + input_t) / 2
                    loss,debug = model.train(sess, model_input)#______训练_____
                    #print(debug[0,:].tolist())
                    step += 1#记录step
                    model_t = time.time() - st - input_t
                    if model_time == 0:#计算模型计算用时
                        model_time = model_t
                    else:
                        model_time = (model_time + model_t) / 2
                    loss_sum += loss
                    if step % FLAGS.log_steps == 0:
                        mean_loss = loss_sum / FLAGS.log_steps
                        print("【train】:mean_loss = %.6f\tbatch = %d\tinput = %.3f sec\ttrain = %.3f sec\tstep = %d\tepochs = %d" \
                              % (mean_loss, FLAGS.batch_size, input_time, model_time, step, n+1))
                        sys.stdout.flush()
                        input_time = 0
                        model_time = 0
                        loss_sum = 0
                    if step % FLAGS.eval_per_num == 0:#训练step达到要求会调用一次test
                        test(sess, test_path, model, n, step)
                    if step % FLAGS.save_iter == 0:#结束训练
                        test(sess, test_path, model, n, step,is_part_test=False) #训练结束，保存
                        export_model(sess, model)
                        return
            except tf.errors.OutOfRangeError:#-------训练数据读取完毕执行test-----------
                if n+1 == FLAGS.train_epochs:#改为训练结束后验证一下
                    test(sess, test_path, model, n, step,is_part_test=False) #训练结束，保存
                    export_model(sess, model)
        sys.stdout.flush()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    def seed_tensorflow(seed=42):
        random.seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = "1"
        np.random.seed(seed)
        tf.set_random_seed(seed)
    seed_tensorflow(100)
    main()

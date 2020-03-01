#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors


import argparse
import itertools
import os
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0,"../..")#插入当前路径
import thumt.data.dataset as dataset
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.search as search
from tensorflow.contrib import crf

def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str,default='../../data/scope/test_src.txt', 
                        help="Path of input file")
    parser.add_argument("--glove_emb_path", type=str, default='../../data/scope/glo_news_50d_ch.txt',
                        help="Path of glove embeddings")
    parser.add_argument("--bert_emb_path", type=str, default=None,
                        help="Path of bert embeddings")
    parser.add_argument("--output", type=str, default="../output/result.txt",
                        help="Path of output file")
    parser.add_argument("--checkpoints", type=str, default="checkpoint",
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, default="../../data/scope/vocab.w ../../data/scope/vocab.t ../../data/scope/vocab.c",
                        help="Path of source and target vocabulary")

    # model and configuration
    parser.add_argument("--models", type=str,  default='rnnsearch',
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="decode_batch_size=1,use_bert=False", 
                        help="Additional hyper parameters")  
    return parser.parse_args()


def default_parameters():
    params = tf.contrib.training.HParams(
        input=None,
        output=None,
        vocabulary=None,
        model=None,
        buffer_size=10000,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        mapping=None,
        append_eos=False,
        gpu_memory_fraction=1,
        bert_size=0,
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=0,
        decode_batch_size=32,
        decode_constant=5.0,
        decode_normalize=False,
        device_list=[0],
        num_threads=6
    )

    return params


def merge_parameters(params1, params2):

    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().items():
        params.add_hparam(k, v)

    params_dict = list(params.values())  ## key value pair

    for (k, v) in params2.values().items():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params



def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_parameters(params, args):
    params.input = args.input
    params.glove_emb_path = args.glove_emb_path
    params.bert_emb_path = args.bert_emb_path
    if args.parameters:
        params.parse(args.parameters)

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(args.vocabulary.split()[0]),
        "target": vocabulary.load_vocabulary(args.vocabulary.split()[1]),
        "char": vocabulary.load_vocabulary(args.vocabulary.split()[2])
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(#这里是否在三个字典中加eos???
        params.vocabulary["source"], params
    )
    params.vocabulary["target"] = vocabulary.process_vocabulary(
        params.vocabulary["target"], params
    )
    params.vocabulary["char"] = vocabulary.process_vocabulary(
        params.vocabulary["char"], params
    )

    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            control_symbols      #看看pad,bos,eos,unk是否在三个字典中，在的话得到序号
        ),
        "target": vocabulary.get_control_mapping(
            params.vocabulary["target"],
            control_symbols
        ),
        "char": vocabulary.get_control_mapping(
            params.vocabulary["char"],
            control_symbols
        )
    }

    return params


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str
        config.gpu_options.per_process_gpu_memory_fraction = params.gpu_memory_fraction

    return config


def set_variables(var_list, value_dict, prefix):
    ops = []
    for var in var_list:
        for name in value_dict.keys():
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:  
                #tf.logging.info("restoring %s -> %s" % (name, var.name))
                with tf.device("/cpu:0"):
                    op = tf.assign(var, value_dict[name])
                    ops.append(op)
                break

    return ops


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load configs
    model_cls_list = [models.get_model(args.models)]
    params_list = [default_parameters() for _ in range(len(model_cls_list))]
    params_list = [
        merge_parameters(params, model_cls.get_parameters())
        for params, model_cls in zip(params_list, model_cls_list)
    ]
    params_list = [
        import_params(args.checkpoints, args.models, params_list[0])#导入训练产生的配置文件
        #for i in range(len([args.checkpoints]))
    ]
    params_list = [
        override_parameters(params_list[i], args)
        for i in range(len(model_cls_list))
    ]

    # Build Graph
    with tf.Graph().as_default():
        model_var_lists = []

        # Load checkpoints
        for i,checkpoint in enumerate([args.checkpoints]):
            print("Loading %s" % checkpoint)
            var_list = tf.train.list_variables(checkpoint)#所有模型变量取成列表
            values = {}
            reader = tf.train.load_checkpoint(checkpoint)

            for (name, shape) in var_list:
                if not name.startswith(model_cls_list[i].get_name()):#获取所有rnnsearch里不带"losses_avg"的变量
                    continue

                if name.find("losses_avg") >= 0:
                    continue

                tensor = reader.get_tensor(name)#获取成数
                values[name] = tensor 

            model_var_lists.append(values)#获取所有rnnsearch里不带"losses_avg"的变量,数值

        # Build models
        model_fns = []

        for i in range(len([args.checkpoints])):
            name = model_cls_list[i].get_name()
            model = model_cls_list[i](params_list[i], name + "_%d" % i)
            model_fn = model.get_inference_func()#调用模型中的推理功能
            model_fns.append(model_fn)

        params = params_list[0]
        
        #features = dataset.get_inference_input_with_bert(args.input, params)
        if params.use_bert and params.bert_emb_path:
            features = dataset.get_inference_input_with_bert(params.input + [params.bert_emb_path], params)
        else:
            features = dataset.get_inference_input([params.input], params)

        predictions = search.create_inference_graph(model_fns, features,
                                                    params)

        assign_ops = []

        all_var_list = tf.trainable_variables()

        for i in range(len([args.checkpoints])):
            un_init_var_list = []
            name = model_cls_list[i].get_name()

            for v in all_var_list:
                if v.name.startswith(name + "_%d" % i):
                    un_init_var_list.append(v)

            ops = set_variables(un_init_var_list, model_var_lists[i],
                                name + "_%d" % i)
            assign_ops.extend(ops)

        assign_op = tf.group(*assign_ops)

        sess_creator = tf.train.ChiefSessionCreator(
            config=session_config(params)
        )

        results = []

        # Create session
        with tf.train.MonitoredSession(session_creator=sess_creator) as sess:
            # Restore variables
            sess.run(assign_op)

            while not sess.should_stop():
                results.extend(sess.run(predictions))
                message = "Finished batch %d" % len(results)
                tf.logging.log(tf.logging.INFO, message)
            tar=[]
            with open(params.input,"r") as inputs_f:
                for line in inputs_f:
                    if line.strip()=="O":
                        continue
                    else:
                        tar.extend(line.split(" ")[:-1])
                      
                        
                    
                    
                
                
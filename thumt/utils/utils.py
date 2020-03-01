# coding=utf-8
# Copyright 2018 The THUMT Authors




import types
import numpy as np
import tensorflow as tf

def load_glove(glove_path, words):
    with open(glove_path, "r", encoding="utf-8") as glove_f:
        all_vectors = []
        word_vector_dict = {}
        ok, unk = 0, 0
        for line in glove_f:
            segs = line.strip().split()
            assert len(segs) == 51
            word_vector_dict[segs[0]] = [float(word) for word in segs[1:]]#建立glove的中文词向量表
            
        for word in words:
            if word in word_vector_dict:
                all_vectors.append(word_vector_dict[word])
                ok += 1
            else:
                all_vectors.append(word_vector_dict['<unk>'])
                unk += 1
        print(f"pre-embeded file {glove_path} have {ok} words in dict,{unk} words not in dict")
        return np.asarray(all_vectors, dtype=np.float32)



def session_run(monitored_session, args):
    # Call raw TF session directly
    return monitored_session._tf_sess().run(args)


def zero_variables(variables, name=None):
    ops = []

    for var in variables:
        with tf.device(var.device):
            op = var.assign(tf.zeros(var.shape.as_list()))
        ops.append(op)

    return tf.group(*ops, name=name or "zero_op")


def replicate_variables(variables, device=None):
    new_vars = []

    for var in variables:
        device = device or var.device
        with tf.device(device):
            name = "replicate/" + var.name.split(":")[0]
            new_vars.append(tf.Variable(tf.zeros(var.shape.as_list()),
                                        name=name, trainable=False))

    return new_vars


def collect_gradients(gradients, variables):
    ops = []

    for grad, var in zip(gradients, variables):
        if isinstance(grad, tf.Tensor):
            ops.append(tf.assign_add(var, grad))
        elif isinstance(grad, tf.IndexedSlices):
            ops.append(tf.scatter_add(var, grad.indices, grad.values))
        else:
            print("grad : ", grad, " with type : ", type(grad)) 
    return tf.group(*ops)


def scale_gradients(grads_and_vars, scale):
    scaled_gradients = []
    variables = []

    for grad, var in gradients:
        if isinstance(grad, tf.IndexedSlices):
            slices = tf.IndexedSlices(scale * grad.values, grad.indices)
            scaled_gradients.append(slices)
            variables.append(var)
        elif isinstance(grad, tf.Tensor):
            scaled_gradients.append(scale * grad)
            variables.append(var)
        else:
            pass
        print("grad : ", grad, " with type : ", type(grad))      
 
    return scaled_gradients, variables

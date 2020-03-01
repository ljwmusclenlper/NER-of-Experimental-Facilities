import tensorflow as tf
from six.moves import reduce

def get_filter(shape):
    filters = tf.get_variable("filters",
                              shape=shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    return filters



def masked_conv1d_and_max(input_tensor, weights, filter_num, kernel_size, params):
    """Applies 1d convolution and a masked max-pooling

    Parameters
    ----------
    t : tf.Tensor
        A tensor with at least 3 dimensions [d1, d2, ..., dn-1, dn]
    weights : tf.Tensor of tf.bool return by sequence_mask
        A Tensor of shape [d1, d2, dn-1]
    filter_num : int
        number of filter_num
    kernel_size : int
        kernel size for the temporal convolution
    params : dict
        hyper params 

    Returns
    -------
    tf.Tensor
        A tensor of shape [d1, d2, dn-1, filter_num]

    """
    # Get shape and parameters

    shape = tf.shape(input_tensor)#[b,seq,char,50]
    ndims = input_tensor.shape.ndims
    # dim1 = reduce(lambda x, y: x*y, [shape[i] for i in range(ndims - 2)])
    dim1 = shape[0] * shape[1]
    dim2 = shape[-2]
    dim3 = shape[-1]#[dim1,dim2,dim3]=[字数，每个字字母数，字母特征维度]

    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])#[b,seq,cha]==>[b*seq,chr,1]
    weights = tf.to_float(weights)#[字数，每个字字母数，1]，每个单词不够最长字母个数的地方是0

    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    input_tensor = tf.reshape(input_tensor, shape=flat_shape)
    input_tensor *= weights   
    # ATTENTION: the above op will cause input_tensor (?, ?, ?, 128) => (?, ?, ?)
    #            which will cause tf.layers.conv1d doesn't work
    # t_conv = tf.nn.conv1d(input_tensor, filter_num, kernel_size, padding='same')

    filter_shape = [kernel_size, params.char_embedding_size, filter_num]#[3,50,50]
    filters = get_filter(filter_shape)#初始化卷积核权重值

    t_conv = tf.nn.conv1d(
        input_tensor,#[整个batch字数，每个字最大char数,50]
        filters,
        stride=1,
        padding="SAME",
        name="conv"
    )#[字数，1,每个字字母数，50]====》[1,3,50,50]====》squeeze_dim==>[字数，每个字字母数,50]

    t_max = tf.reduce_max(t_conv, axis=-2)#用一个字中最大特征值的字母的特征代表一个字[字数,50]

    # # Reshape the output
    final_shape = [shape[i] for i in range(ndims-2)] + [filter_num] # [batch, seq_len, filter_num]
    t_max = tf.reshape(t_max, shape=final_shape)
    return t_max

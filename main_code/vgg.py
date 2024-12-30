import numpy as np
from scipy.io import matlab
import tensorflow as tf
# from tensorflow.keras.applications import VGG19
# from tensorflow.keras.models import Model
# from functools import reduce

# from tensorflow.keras.applications.vgg19 import preprocess_input # 这个函数是专门为 VGG
# 等神经网络设计的，它会将图像进行标准化，使其能够与预训练模型的训练数据相匹配，这种预处理通常会将像素值从 [0, 255] 的范围映射到 [-1, 1] 或 [0, 1] 的范围，具体取决于特定模型的要求 from PIL
# import Image
# 获取可见的 GPU 设备列表

gpus = tf.config.experimental.list_physical_devices('GPU')

# print("TensorFlow 可见的 GPU 设备:", gpus)

# 打印默认设备（如果使用了默认设备）
# default_device = tf.test.gpu_device_name()
# print("默认设备:", default_device)

# 告诉TensorFlow在需要时动态分配GPU内存，而不是预先分配所有可用的GPU内存

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
)


def load_net(data_path):
    data = matlab.loadmat(data_path)
    if "normalization" in data:
        # old format, for data where
        # MD5(imagenet-vgg-verydeep-19.mat) = 8ee3263992981a1d26e73b3ca028a123
        mean_pixel = np.mean(data["normalization"][0][0][0], axis=(0, 1))
    else:
        # new format, for data where
        # MD5(imagenet-vgg-verydeep-19.mat) = 106118b7cf60435e6d8e04f6a6dc3657
        mean_pixel = data["meta"]["normalization"][0][0][0][0][2][0][0]
    weights = data["layers"][0]
    return weights, mean_pixel


def net_preloaded(weights, input_image, pooling):
    net = {}
    current = input_image
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == "conv":
            if isinstance(weights[i][0][0][0][0], np.ndarray):
                # old format
                kernels, bias = weights[i][0][0][0][0]
            else:
                # new format
                kernels, bias = weights[i][0][0][2][0]

            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == "relu":
            current = tf.nn.relu(current)
        elif kind == "pool":
            current = _pool_layer(current, pooling)
        net[name] = current

    assert len(net) == len(VGG19_LAYERS)
    return net


def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding="SAME")
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input, pooling):
    if pooling == "avg":
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")





def preprocess(image):
    mean_pixel = np.array([123.68, 116.779, 103.939])
    image = tf.cast(image, tf.float32)
    # 对图像数组的三个通道分别减去均值像素
    channels = tf.unstack(image, axis=-1)  # 将图像张量按通道分解
    for i in range(3):  # 对于每个颜色通道
        channels[i] -= mean_pixel[i]  # 减去均值像素值
    image = tf.stack(channels, axis=-1)  # 重新堆叠通道以得到图像张量

    return image


def unprocess(image):
    mean_pixel = np.array([123.68, 116.779, 103.939])
    image = tf.cast(image, tf.float32)
    channels = tf.unstack(image, axis=-1)  # 将图像张量按通道分解
    for i in range(3):  # 对于每个颜色通道
        channels[i] += mean_pixel[i]  # 减去均值像素值
    image = tf.stack(channels, axis=-1)  # 重新堆叠通道以得到图像张量

    return image


def _tensor_size(tensor):
    return tf.size(tensor).numpy()  # 使用 tf.size() 方法计算张量大小


def calculate_content_loss(vgg_weights, image, pooling, content_features, content_weight, content_layers_weights,
                           CONTENT_LAYERS):
    net = net_preloaded(vgg_weights, image, pooling)
    content_loss = 0
    content_losses = []
    for content_layer in CONTENT_LAYERS:
        content_losses.append(
            content_layers_weights[content_layer] * content_weight
            * (
                    2  # "gradient normalization"（梯度归一化）
                    * tf.nn.l2_loss(net[content_layer] - content_features[content_layer])
                    / _tensor_size(content_features[content_layer])
            )
        )
    content_loss += tf.reduce_sum(content_losses)
    return content_loss


def calculate_style_loss(vgg_weights, image, pooling, style_layers_weights, style_features, styles, style_weight,
                         style_blend_weights, STYLE_LAYERS):
    net = net_preloaded(vgg_weights, image, pooling)
    style_loss = 0
    for i in range(len(styles)):
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            _, height, width, number = tf.shape(layer)

            size = tf.size(layer)
            feats = tf.cast(layer, dtype=tf.float32)
            feats = tf.reshape(feats, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / tf.cast(size, tf.float32)

            style_gram = style_features[i][style_layer]
            style_gram_tensor = tf.convert_to_tensor(style_gram, dtype=tf.float32)  # 张量

            style_losses.append(
                style_layers_weights[style_layer]
                * 2
                * tf.nn.l2_loss(gram - style_gram_tensor)
                / _tensor_size(style_gram_tensor)
            )
        style_loss += style_weight * style_blend_weights[i] * tf.reduce_sum(style_losses)

    return style_loss


def tv_loss(image, tv_weight, shape):
    tv_y_size = _tensor_size(image[:, 1:, :, :])
    tv_x_size = _tensor_size(image[:, :, 1:, :])
    tv_loss = (
            tv_weight
            * 2
            * (
                    (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, : shape[1] - 1, :, :]) / tv_y_size)
                    + (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, : shape[2] - 1, :]) / tv_x_size)
            )
    )

    return tv_loss


'''
def content_features(content_array,CONTENT_LAYERS):
    # 进行预处理操作
    content_preprocessed = preprocess(content_array)
    # 定义内容图像的张量
    content_tensor = tf.convert_to_tensor(content_preprocessed, dtype=tf.float32)
    # 在最前面添加一个维度
    content_1 = tf.expand_dims(content_tensor, axis=0)

    # 导入 VGG19 模型，不包括顶部的分类层
    vgg_model = VGG19(weights='imagenet', include_top=False)
    # 获取 VGG19 模型中指定层的输出
    outputs = [vgg_model.get_layer(layer).output for layer in CONTENT_LAYERS]
    # 创建新模型以输出指定层的特征
    feature_extraction_model = Model(inputs=vgg_model.input, outputs=outputs)
    # 获取内容图像在神经网络中各层的特征
    content_features_outputs = feature_extraction_model(content_1)
    for i, layer in enumerate(CONTENT_LAYERS):
        content_features[layer] = content_features_outputs[i].numpy()
    return content_features
'''

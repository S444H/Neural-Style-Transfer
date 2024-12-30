import vgg
import numpy as np
import tensorflow as tf

# from collections import OrderedDict
import os
from functools import reduce
import time
from PIL import Image

CONTENT_LAYERS = ("relu4_2", "relu5_2")
STYLE_LAYERS = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")


def get_loss_vals(loss_store):
    # return OrderedDict((key, val.eval()) for key, val in loss_store
    return dict((key, val.numpy()) for key, val in loss_store.items())


def print_progress(loss_vals):
    for key, val in loss_vals.items():
        print("{:>13s} {:g}".format(key + " loss:", val))


def _tensor_size(tensor):
    from operator import mul

    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb


def hms(seconds):
    seconds = int(seconds)
    hours = seconds // (60 * 60)
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    if hours > 0:
        return "%d hr %d min" % (hours, minutes)
    elif minutes > 0:
        return "%d min %d sec" % (minutes, seconds)
    else:
        return "%d sec" % seconds


"""

    This function yields tuples (iteration, image, loss_vals) at every
    iteration. However `image` and `loss_vals` are None by default. Each `checkpoint_iterations`,    `image` is not None. 
     Each    `print_iterations`,      `loss_vals` is not None.

    `loss_vals` is a dict with loss values for the current iteration, e.g. :
    {'content': 1.23, 'style': 4.56, 'tv': 7.89, 'total': 13.68}


    """


def stylize(
        network,  # 神经网络的配置文件路径
        initial,  # 初始图像
        initial_noiseblend,  # 用来混合初始噪声的比例
        content,  # 内容图像
        styles,  # 风格图像s

        preserve_colors,  # 是否保持原始内容图像的亮度信息
        iterations,  # 迭代次数

        content_weight,
        content_weight_blend,  # 权重混合因子，用于平衡两个选定内容层的权重
        style_weight,
        style_layer_weight_exp,  # 指数级增加不同风格层的权重，这使得更高层的特征对风格重建的贡献更大
        style_blend_weights,  # 风格层权重的混合权重，用于平衡多个选定风格的权重
        tv_weight,  # 总变差损失的权重，总变差损失用于确保生成的图像平滑连续，避免出现噪点和过度纹理

        learning_rate,  # 优化器的学习率
        beta1,
        beta2,  # Adam 优化器的超参数，用于调整梯度的指数衰减率和二次方梯度的指数衰减率
        epsilon,  # Adam 优化器的数值稳定性参数，用于防止除零错误
        pooling,  # 池化层的类型
        print_iterations=None,
        checkpoint_iterations=None,

):
    '''
    对于图像数据，一般的形状表示方式是 (batch_size, height, width, channels)，其中 batch_size 是批量大小，而 height、width 和 channels 分别表示图像的高度、宽度和通道数。
    如 content.shape 是一个包含内容图像维度信息的元组，比如 (height, width, channels)，那么这个形状元组将变成 (1, height, width, channels)。在最前面增加了一个额外的维度，这个维度大小为 1，代表一个单独的样本或者批量中的第一个样本。
    '''
    # 图片信息
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    '''使用 Eager Execution 模式'''

    # 在前向传播模式下提取内容图像在神经网络中各层的特征
    content_features = {}  # 存储内容图像在神经网络中各层的特征值
    # 进行预处理操作
    content_preprocessed = vgg.preprocess(content)
    # 定义内容图像的张量
    content_tensor = tf.convert_to_tensor(content_preprocessed, dtype=tf.float32)
    # 在最前面添加一个维度
    content_1 = tf.expand_dims(content_tensor, axis=0)
    net = vgg.net_preloaded(vgg_weights, content_1,
                            pooling)  # ---------------------------------------------------------------------------
    # 获取特定层的输出
    for layer_name in CONTENT_LAYERS:
        content_features[layer_name] = net[layer_name]
    # keras 通过模块接口使用VGG19模型
    '''
    content_features = {} 
    content_features = vgg.content_features(content)
    '''

    # 在前向传播模式下提取style图像s在神经网络中各层的特征
    style_features = [{} for _ in styles]  # 列表中包含了与样式图像数量相同的字典,每个字典用于存储对应样式图像的不同层的特征信息
    for i in range(len(styles)):
        # 进行预处理操作
        style_preprocessed = vgg.preprocess(styles[i])
        # 定义内容图像的张量
        style_tensor = tf.convert_to_tensor(style_preprocessed, dtype=tf.float32)
        # 在最前面添加一个维度
        style_1 = tf.expand_dims(style_tensor, axis=0)
        net = vgg.net_preloaded(vgg_weights, style_1,
                                pooling)  # ---------------------------------------------------------------------------
        for layer_name in STYLE_LAYERS:
            feature = net[layer_name]

            feature = np.reshape(feature,
                                 (-1, feature.shape[3]))  # 将特征张量 features 在第四个维度（RGB）上的数据重新整形为一个新的二维数组用于 gram 矩阵计算
            gram = np.matmul(feature.T, feature) / feature.size

            style_features[i][layer_name] = gram

    # 使用反向传播（backpropagation）制作风格化图像
    initial_content_noise_coeff = 1.0 - initial_noiseblend
    if initial is None:
        noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
        initial = tf.random.normal(shape) * 0.256
    else:
        initial = vgg.preprocess(initial)
        initial = tf.convert_to_tensor(initial, dtype=tf.float32)
        initial = tf.expand_dims(initial, axis=0)
        initial = initial * initial_content_noise_coeff + (tf.random.normal(shape) * 0.256) * (
                    1.0 - initial_content_noise_coeff)
    image = tf.Variable(initial)  # 创建可变的张量，即变量,后续的操作中使用 image 这个变量进行图像处理、优化或者其他相关的操作
    # net = vgg.net_preloaded(vgg_weights, image, pooling) #---------------------------------------------------------------------------

    '''求内容损失'''
    # 定义内容损失（content loss）中不同层的权重
    content_layers_weights = {"relu4_2": content_weight_blend, "relu5_2": 1.0 - content_weight_blend}
    # content_loss = vgg.calculate_content_loss(vgg_weights, image, pooling, content_features, content_weight, content_layers_weights, CONTENT_LAYERS)
    '''
    # 定义内容损失（content loss）中不同层的权重
    content_layers_weights = {}
    content_layers_weights["relu4_2"] = content_weight_blend
    content_layers_weights["relu5_2"] = 1.0 - content_weight_blend

    content_loss = 0
    content_losses = []
    for content_layer in CONTENT_LAYERS:
        content_losses.append(
                content_layers_weights[content_layer] * content_weight
                * (
                    2  #  "gradient normalization"（梯度归一化），通过乘以一个常数（如这里的 2），可以调整损失函数的梯度，从而影响优化算法的收敛速度和稳定性，具体的常数值可能是根据实验或经验得出的，用于调整优化过程中的梯度
                    * tf.nn.l2_loss(net[content_layer] - content_features[content_layer])  # 使用了 L2 范数（欧几里得距离）来衡量生成图像和目标内容图像之间的差异
                    / _tensor_size(content_features[content_layer])
                  )
                             )
    content_loss += reduce(tf.add, content_losses)
    '''

    '''求风格损失'''
    # 使用 style_layer_weight_exp 定义style_layer各层weight，并进行归一占比（normalize）
    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp
    # normalize style layer weights ，指数级增加不同风格层的权重，这使得更高层的特征对风格重建的贡献更大
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum

    # style_loss = vgg.calculate_style_loss(vgg_weights, image, pooling, style_layers_weights, style_features, styles, style_weight, style_blend_weights, STYLE_LAYERS)

    '''
    style_loss = 0
    for i in range(len(styles)):
        style_losses = []
        for style_layer in STYLE_LAYERS:

            layer = net[style_layer]
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            size = height * width * number

            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size

            style_gram = style_features[i][style_layer]
            style_gram_tensor = tf.convert_to_tensor(style_gram, dtype=tf.float32) # 张量

            style_losses.append(
                    style_layers_weights[style_layer]
                    * 2
                    * tf.nn.l2_loss(gram - style_gram_tensor)
                    / _tensor_size(style_gram_tensor)
                               )
        style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)
        '''

    '''总变差损失'''
    # 总变差去噪:用于减少图像噪声和保持图像细节
    '''
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
    '''
    # tv_loss = vgg.tv_loss(image, tv_weight, shape)

    # optimizer setup
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta1, beta2, epsilon)
    # optimization
    best_loss = float("inf")
    best = None
    iteration_times = []
    start = time.time()

    print("Optimization started...")

    for i in range(iterations):
        iteration_start = time.time()
        if i > 0:
            elapsed = time.time() - start
            remaining = np.mean(iteration_times[-10:]) * (iterations - i)

            print('第{0}次迭代，已训练时间：{1}，预计还剩时间：{2}，total_loss：{3}'.format(i + 1, hms(elapsed), hms(remaining),loss))

        else:
            print('第{0}次迭代，共需迭代数：{1}'.format(i + 1, iterations))

        with tf.GradientTape() as tape:
            content_loss = vgg.calculate_content_loss(vgg_weights, image, pooling, content_features, content_weight,
                                                      content_layers_weights, CONTENT_LAYERS)
            style_loss = vgg.calculate_style_loss(vgg_weights, image, pooling, style_layers_weights, style_features,
                                                  styles, style_weight,
                                                  style_blend_weights, STYLE_LAYERS)
            tv_loss = vgg.tv_loss(image, tv_weight, shape)
            loss = content_loss + style_loss + tv_loss
            loss_store = dict([("content", content_loss), ("style", style_loss), ("tv", tv_loss), ("total", loss)])

        # 计算损失相对于变量（这里是图像）的梯度
        gradients = tape.gradient(loss, image)
        gradients = tf.expand_dims(gradients, axis=0)
        # 应用梯度更新图像
        optimizer.apply_gradients(zip(gradients, [image]))

        # 控制何时打印完整损失值进度信息，默认为最后一次迭代输出完整信息
        last_step = (i == iterations - 1)
        if last_step or (print_iterations and i % print_iterations == 0):
            loss_vals = get_loss_vals(loss_store)
            print_progress(loss_vals)
        else:
            loss_vals = None

        # 是否达到检查点,定期保存训练过程中的状态
        if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
            this_loss = loss
            if this_loss < best_loss:
                best_loss = this_loss
                best = image
            img_out = vgg.unprocess(tf.reshape(best, shape[1:]))

            if preserve_colors:
                original_image = np.clip(content, 0, 255)
                styled_image = np.clip(img_out, 0, 255)

                # 保持原始内容图像的亮度信息，而使用风格化图像的色彩和纹理信息
                # Luminosity transfer steps:
                # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                # 2. Convert stylized grayscale into YUV (YCbCr)
                # 3. Convert original image into YUV (YCbCr)
                # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                # 5. Convert recombined image from YUV back to RGB

                # 1
                styled_grayscale = rgb2gray(styled_image)
                styled_grayscale_rgb = gray2rgb(styled_grayscale)
                # 2
                styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert("YCbCr"))
                # 3
                original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert("YCbCr"))
                # 4
                w, h, _ = original_image.shape
                combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                combined_yuv[..., 1] = original_yuv[..., 1]
                combined_yuv[..., 2] = original_yuv[..., 2]
                # 5
                img_out = np.array(Image.fromarray(combined_yuv, "YCbCr").convert("RGB"))

        else:
            img_out = None
        yield i + 1 if last_step else i, img_out, loss_store

        iteration_end = time.time()
        iteration_times.append(iteration_end - iteration_start)



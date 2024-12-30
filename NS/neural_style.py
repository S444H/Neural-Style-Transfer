import os
import math
import re
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser  # 用于解析命令行参数,简化了为脚本定义参数和选项的过程，并自动生成帮助消息
# from collections import OrderedDict
from stylize import stylize

# default arguments
ITERATIONS = 1000

CONTENT_WEIGHT = 5e0               # 默认 5
CONTENT_WEIGHT_BLEND = 0.5         # 默认 1
STYLE_WEIGHT = 5e2                 # 默认 500
TV_WEIGHT = 1e2                    # 默认 100
STYLE_LAYER_WEIGHT_EXP = 1         # 默认 1

LEARNING_RATE = 10                # 默认 10
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08

STYLE_SCALE = 1.0

VGG_PATH = "../imagenet-vgg-verydeep-19.mat"
POOLING = "max"
OUTPUT = "./output/1.png"


def imread(path):
    img = np.array(Image.open(path)).astype(np.float32)
    if len(img.shape) == 2:
        # 若图像是灰度图（即通道数为 2），则将其转换为 3 通道的灰度图
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        # 若图像是带有 Alpha 通道的 PNG 图像（即通道数为 4），则丢弃 Alpha 通道，只保留 RGB 通道
        img = img[:, :, :3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


def imresize(arr, size):
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if isinstance(size, tuple):
        height, width = size
    else:
        width = int(img.width * size)
        height = int(img.height * size)
    return np.array(img.resize((width, height)))


def fmt_imsave(fmt, iteration):
    if re.match(r"^.*\{.*\}.*$", fmt):
        return fmt.format(iteration)
    elif "%" in fmt:
        return fmt % iteration
    else:
        raise ValueError("illegal format string '{}'".format(fmt))


def build_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--content", dest="content", help="content image", metavar="CONTENT", required=True
    )
    parser.add_argument(
        "--styles",
        dest="styles",
        nargs="+",
        help="one or more style images",
        metavar="STYLE",
        required=True,
    )

    parser.add_argument(
        "--output", dest="output", help="output path", metavar="OUTPUT", default=OUTPUT,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        dest="iterations",
        help="iterations (default %(default)s)",
        metavar="ITERATIONS",
        default=ITERATIONS,
    )

    parser.add_argument(
        "--print-iterations",
        type=int,
        dest="print_iterations",
        help="statistics printing frequency",
        metavar="PRINT_ITERATIONS",
        default=None,
    )
    parser.add_argument(
        "--checkpoint-output",
        dest="checkpoint_output",
        help="checkpoint output format, e.g. output_{:05}.jpg or " "output_%%05d.jpg",
        metavar="OUTPUT",
        default=None,
    )
    parser.add_argument(
        "--checkpoint-iterations",
        type=int,
        dest="checkpoint_iterations",
        help="checkpoint frequency",
        metavar="CHECKPOINT_ITERATIONS",
        default=None,
    )

    parser.add_argument(
        "--progress-write",
        default=False,
        action="store_true",
        help="write iteration progess data to OUTPUT's dir",
        required=False,
    )
    parser.add_argument(
        "--progress-plot",
        default=False,
        action="store_true",
        help="plot iteration progess data to OUTPUT's dir",
        required=False,
    )

    parser.add_argument("--width", type=int, dest="width", help="output width", metavar="WIDTH")
    parser.add_argument(
        "--style-scales",
        type=float,
        dest="style_scales",
        nargs="+",
        help="one or more style scales",
        metavar="STYLE_SCALE",
    )

    parser.add_argument(
        "--network",
        dest="network",
        help="path to network parameters (default %(default)s)",
        metavar="VGG_PATH",
        default=VGG_PATH,
    )
    parser.add_argument(
        "--content-weight-blend",
        type=float,
        dest="content_weight_blend",
        help="content weight blend, conv4_2 * blend + conv5_2 * (1-blend) " "(default %(default)s)",
        metavar="CONTENT_WEIGHT_BLEND",
        default=CONTENT_WEIGHT_BLEND,
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        dest="content_weight",
        help="content weight (default %(default)s)",
        metavar="CONTENT_WEIGHT",
        default=CONTENT_WEIGHT,
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        dest="style_weight",
        help="style weight (default %(default)s)",
        metavar="STYLE_WEIGHT",
        default=STYLE_WEIGHT,
    )
    parser.add_argument(
        "--style-layer-weight-exp",
        type=float,
        dest="style_layer_weight_exp",
        help="style layer weight exponentional increase - "
             "weight(layer<n+1>) = weight_exp*weight(layer<n>) "
             "(default %(default)s)",
        metavar="STYLE_LAYER_WEIGHT_EXP",
        default=STYLE_LAYER_WEIGHT_EXP,
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        dest="tv_weight",
        help="total variation regularization weight (default %(default)s)",
        metavar="TV_WEIGHT",
        default=TV_WEIGHT,
    )

    parser.add_argument(
        "--style-blend-weights",
        type=float,
        dest="style_blend_weights",
        help="style blending weights",
        nargs="+",
        metavar="STYLE_BLEND_WEIGHT",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        dest="learning_rate",
        help="learning rate (default %(default)s)",
        metavar="LEARNING_RATE",
        default=LEARNING_RATE,
    )
    parser.add_argument(
        "--beta1",
        type=float,
        dest="beta1",
        help="Adam: beta1 parameter (default %(default)s)",
        metavar="BETA1",
        default=BETA1,
    )
    parser.add_argument(
        "--beta2",
        type=float,
        dest="beta2",
        help="Adam: beta2 parameter (default %(default)s)",
        metavar="BETA2",
        default=BETA2,
    )
    parser.add_argument(
        "--eps",
        type=float,
        dest="epsilon",
        help="Adam: epsilon parameter (default %(default)s)",
        metavar="EPSILON",
        default=EPSILON,
    )

    parser.add_argument("--initial", dest="initial", help="initial image", metavar="INITIAL")
    parser.add_argument(
        "--initial-noiseblend",
        type=float,
        dest="initial_noiseblend",
        help="ratio of blending initial image with normalized noise "
             "(if no initial image specified, content image is used) "
             "(default %(default)s)",
        metavar="INITIAL_NOISEBLEND",
    )
    parser.add_argument(
        "--preserve-colors",
        action="store_true",
        dest="preserve_colors",
        help="style-only transfer (preserving colors) - if color transfer " "is not needed",
    )
    parser.add_argument(
        "--pooling",
        dest="pooling",
        help="pooling layer configuration: max or avg (default %(default)s)",
        metavar="POOLING",
        default=POOLING,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        dest="overwrite",
        help="write file even if there is already a file with that name",
    )
    return parser


# 主函数，程序的入口
def main():
    # 设置 TensorFlow 的日志级别为最低,只显示错误信息，而不显示其他信息,用于减少输出，以使输出信息更加简洁
    key = "TF_CPP_MIN_LOG_LEVEL"
    if key not in os.environ:
        os.environ[key] = "2"

    # 创建命令行参数解析器
    parser = build_parser()
    options = parser.parse_args()

    # 检查网络参数文件是否存在
    if not os.path.isfile(options.network):
        parser.error(
            "Network %s does not exist." % options.network
        )


    # 检查输出路径是否可写，并尝试保存一个虚拟图像
    if os.path.isfile(options.output) and not options.overwrite:
        raise IOError(
            "%s already exists, will not replace it without "
            "the '--overwrite' flag" % options.output
        )
    try:
        imsave(options.output, np.zeros((500, 500, 3)))
    except:
        raise IOError(
            "%s is not writable or does not have a valid file "
            "extension for an image file" % options.output
        )


    # 检查 checkpoint_output 和 checkpoint_iterations 参数的合法性
    # 这两个参数是相关联的，必须同时使用或者同时不使用，以确保正确性
    if [options.checkpoint_iterations, options.checkpoint_output].count(None) == 1:
        parser.error('"checkpoint_output" and "checkpoint_iterations"必须同时使用或者同时不使用.')

    # 检查保存中间图像的格式化字符串是否合法
    if options.checkpoint_output is not None:
        if re.match(r"^.*(\{.*\}|%.*).*$", options.checkpoint_output) is None:
            parser.error(
                "To save intermediate images, the checkpoint_output "
                "parameter must contain placeholders (e.g. "
                "`foo_{}.jpg` or `foo_%d.jpg`"
            )

    # 读取内容图像和风格图像，并根据给定的参数进行预处理
    content_image = imread(options.content)
    style_images = [imread(style) for style in options.styles]

    # 根据输入宽度调整内容图像的尺寸
    width = options.width
    if width is not None:
        new_shape = (
            int(math.floor(float(content_image.shape[0]) / content_image.shape[1] * width)),
            width,
        )
        content_image = imresize(content_image, new_shape)
    target_shape = content_image.shape

    # 根据风格比例和内容图像的尺寸对风格图像进行调整
    for i in range(len(style_images)):
        style_scale = STYLE_SCALE
        if options.style_scales is not None:
            style_scale = options.style_scales[i]
        style_images[i] = imresize(style_images[i], style_scale * target_shape[1] / style_images[i].shape[1])

    # 根据风格权重，得出 style_blend_weights 参数
    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # 默认风格权重相等
        style_blend_weights = [1.0 / len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight / total_blend_weight for weight in style_blend_weights]

    # 设置初始图像
    initial = options.initial
    if initial is not None:
        initial = imresize(imread(initial), content_image.shape[:2])
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 0.0
    else:
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 1.0    # 此时这个数值没有意义，因为在代码逻辑当中只是为了使是一个数值
        if options.initial_noiseblend < 1.0:
            initial = content_image



    loss_arrs = None

    # 执行style迁移算法，并保存中间图像和损失值信息
    for iteration, image, loss_vals in stylize(
            network=options.network,
            initial=initial,
            initial_noiseblend=options.initial_noiseblend,
            content=content_image,
            styles=style_images,
            preserve_colors=options.preserve_colors,
            iterations=options.iterations,

            content_weight=options.content_weight,
            content_weight_blend=options.content_weight_blend,
            style_weight=options.style_weight,
            style_layer_weight_exp=options.style_layer_weight_exp,
            style_blend_weights=style_blend_weights,
            tv_weight=options.tv_weight,

            learning_rate=options.learning_rate,
            beta1=options.beta1,
            beta2=options.beta2,
            epsilon=options.epsilon,
            pooling=options.pooling,

            print_iterations=options.print_iterations,
            checkpoint_iterations=options.checkpoint_iterations,
    ):
        if (image is not None) and (options.checkpoint_output is not None):
            imsave(fmt_imsave(options.checkpoint_output, iteration), image)
        if (loss_vals is not None) and (options.progress_plot or options.progress_write):
            if loss_arrs is None:
                itr = []
                loss_arrs = dict((key, []) for key in loss_vals.keys())
            for key, val in loss_vals.items():
                loss_arrs[key].append(val)
            itr.append(iteration)

    # 保存最终图像
    imsave(options.output, image)

    '''
    # 如果设置了保存损失值信息的选项，将损失值信息保存为文本文件
    if options.progress_write:
        fn = "{}/progress.txt".format(os.path.dirname(options.output))
        tmp = np.empty((len(itr), len(loss_arrs) + 1), dtype=float)
        tmp[:, 0] = np.array(itr)
        for ii, val in enumerate(loss_arrs.values()):
            tmp[:, ii + 1] = np.array(val)
        np.savetxt(fn, tmp, header=" ".join(["itr"] + list(loss_arrs.keys())))
    '''


    # 如果设置了绘制损失曲线的选项，将损失曲线绘制并保存为图片
    if options.progress_plot:
        import matplotlib
        matplotlib.use("Agg")
        fig, ax = plt.subplots()
        for key, val in loss_arrs.items():
            ax.semilogy(itr, val, label=key)
        ax.legend()
        ax.set_xlabel("iterations")
        ax.set_ylabel("loss")
        fig.savefig("{}/progress.png".format(os.path.dirname(options.output)))


if __name__ == "__main__":
    main()

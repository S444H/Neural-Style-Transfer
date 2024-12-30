from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description="一个使用argparse的简单脚本")
    parser.add_argument('input_file', help='输入文件的路径')
    parser.add_argument('-o', '--output', help='输出文件的路径')

    args = parser.parse_args()

    # 获取用户提供的数值
    input_file = args.input_file
    output_file = args.output

    print(f"输入文件: {input_file}")
    if output_file:
        print(f"输出文件: {output_file}")
    else:
        print("未指定输出文件")

if __name__ == "__main__":
    main()
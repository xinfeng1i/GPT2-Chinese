# coding: utf-8
#
# Author: xinfengli
# Date: 2019/09/02

"""
从文本文件中构建 GPT2 的训练数据.
"""

import torch
import os
import json
import argparse
from tqdm import tqdm
from tokenizations.bpe_tokenizer import get_encoder

import logging
logging.basicConfig(level=logging.INFO)


def build_files(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)
        lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
        all_len = len(lines)
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]

        # 将多余的部分增加到最后一个chunk
        if i == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
        sublines = [full_tokenizer.tokenize(line) for line in sublines if
                    len(line) > min_length]  # 只考虑长度超过min_length的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]

        # `full_line` 记录 Ids 列表
        full_line = []
        for subline in sublines:
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 文章开头添加MASK表示文章开始
            full_line.extend(subline)
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 文章之间添加CLS表示文章结束
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
    print('finish')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', type=str, required=True, help='选择词库')
    parser.add_argument('--raw_data_path', type=str, required=True, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', type=str, required=True, help='tokenized语料存放位置')
    parser.add_argument('--num_pieces', default=1, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=10, type=int, required=False, help='最短收录文章长度')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    from tokenizations import tokenization_bert
    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    print('building files...')
    build_files(data_path=args.raw_data_path, tokenized_data_path=args.tokenized_data_path, num_pieces=args.num_pieces,
                full_tokenizer=full_tokenizer, min_length=args.min_length)
    print('files built')


if __name__ == "__main__":
    main()
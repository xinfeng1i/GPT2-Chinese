# coding: utf-8
#
# Author: xinfengli
# Date: 2019/09/20


"""
将 BERT 格式的文本数据转化为 GPT2 训练所需要的 Token Ids.

BERT 格式的文本数据为:
   One sentence per line
   Empty line is inserted in different documents.
"""

from tokenizations import tokenization_bert
from tqdm import tqdm
import argparse
import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert(infile, outfile, tokenizer):
    current_doc = []
    with open(infile, "r", encoding="utf-8") as fr, open(outfile, "w", encoding="utf-8") as fw:
        for line in tqdm(fr, desc="Loading Dataset", unit=" lines"):
            if line.strip() != "":
                current_doc.append(line.strip())
            else:
                one_doc = "\n".join(current_doc)
                if one_doc.strip() != "":
                    # 同一个 doc 的不同句子之间以 [SEP] 分隔
                    article = one_doc.strip().replace("\n", " [SEP] ")
                    tokens = tokenizer.tokenize(article)
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)

                    full_input_ids = []
                    full_input_ids.append(tokenizer.convert_tokens_to_ids("[MASK]"))
                    full_input_ids.extend(input_ids)
                    full_input_ids.append(tokenizer.convert_tokens_to_ids("[CLS]"))

                    for id in full_input_ids:
                        fw.write(str(id) + " ")

                current_doc = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, default="cache/vocab.txt", required=False, help="词表路径")
    parser.add_argument("--input_file", type=str, required=True, help="输入文本文件, 格式: 每个句子占一行，不同 doc 之间以空行隔开")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件, 格式：Token Ids")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    # parameter checking
    if not os.path.exists(args.input_file) or not os.path.isfile(args.input_file) \
            or not os.access(args.input_file, os.R_OK):
        logger.error("Input file [%s] not exists or not readable!" % args.input_file)
        sys.exit(1)

    if not os.path.exists(args.vocab_file) or not os.path.isfile(args.vocab_file) \
            or not os.access(args.vocab_file, os.R_OK):
        logger.error("Vocab file [%s] not exists or not readable!" % args.vocab_file)
        sys.exit(1)

    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.vocab_file)
    convert(args.input_file, args.output_file, full_tokenizer)

    logger.info("Data convert finished!")


if __name__ == "__main__":
    main()


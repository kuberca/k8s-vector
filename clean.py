import re
import argparse
import os

parser = argparse.ArgumentParser(description='log file .')
parser.add_argument('log_file', metavar='log_file', type=str, help='log_file file')
args = parser.parse_args()
log_file = args.log_file

file = open(log_file, 'r')
content = file.read()
out = re.split('\\| |:|;|,|\*|\"|\'|=|\[|\]|\(|\)|/|{|}' ,content)
with open(log_file + ".clean", "w") as f:
    f.write(" ".join(out))


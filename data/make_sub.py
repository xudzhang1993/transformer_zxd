import codecs
from collections import Counter
import json
def generate_sub(data_file, sub_file, num):
    counter = Counter()
    with codecs.open(data_file, 'r', 'utf-8') as f: 
        with codecs.open(sub_file, 'w', 'utf-8') as fw:
            for i, line in enumerate(f):
                fw.write(line)
                if i == num - 1:
                    break


if __name__ == "__main__":
    generate_sub("./clean.data", "./sub_100.data", 100)
    



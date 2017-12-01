import codecs
from collections import Counter
import json
def generate_vocab(data_file, vocab_file):
    counter = Counter()
    with codecs.open(data_file, 'r', 'utf-8') as f: 
        for i, line in enumerate(f):
            line = line.strip()
            post, response = line.split('\t')
            post = post.split(' ')
            response = response.split(' ')
            counter.update(post)
            counter.update(response)
            if not i % 100000:
                print "have processed %d lines" % i
        print "total lines: %d" % i
    del counter['']
    counter_1w = counter.most_common(10000)
    counter_2w = counter.most_common(20000)
    counter_all = counter.most_common()
    with codecs.open(vocab_file + '_1w.data', 'w', 'utf-8') as f:
        f.write('<PAD>' + '\n')
        f.write('<UNK>' + '\n')
        f.write('<SOS>' + '\n')
        f.write('<EOS>' + '\n')
        for word, num in counter_1w:
            f.write(word + '\n')

    with codecs.open(vocab_file + '_2w.data', 'w', 'utf-8') as f:
        f.write('<PAD>' + '\n')
        f.write('<UNK>' + '\n')
        f.write('<SOS>' + '\n')
        f.write('<EOS>' + '\n')
        for word, num in counter_2w:
            f.write(word + '\n')

    with codecs.open(vocab_file + '_all.data', 'w', 'utf-8') as f:
        f.write('<PAD>' + '\n')
        f.write('<UNK>' + '\n')
        f.write('<SOS>' + '\n')
        f.write('<EOS>' + '\n')
        for word, num in counter_all:
            f.write(word + '\n')


if __name__ == "__main__":
    generate_vocab('./clean.data', './vocab')

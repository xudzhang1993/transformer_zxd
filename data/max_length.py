import codecs


def max_length(data_file, length_file):
    max_post_len, max_response_len = 0, 0
    with codecs.open(data_file, 'r', 'utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            post, response = line.split('\t')
            post = post.split(' ')
            response = response.split(' ')
            max_post_len = max(len(post), max_post_len)
            max_response_len = max(len(post), max_response_len)
            if i % 100000 == 0:
                print i
    with codecs.open(length_file, 'w', 'utf-8') as f:
        f.write(str(max_post_len) + ' ' + str(max_response_len)) 


if __name__ == "__main__":
    max_length("./clean.data", "./max_length.data") 

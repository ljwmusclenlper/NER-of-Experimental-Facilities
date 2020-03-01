#encoding=utf8
import tensorflow as tf
with tf.Session() as sess:
    mapping_strings = tf.constant(["emerson", "lake", "palmer",])
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping_strings, num_oov_buckets=0, default_value=-1)
    features = tf.constant(["emerson", "lake", "and", "palmer",  'dd'])
    ids = table.lookup(features)
    tf.tables_initializer().run()
    a = ids.eval()
    print(a)
def to_word(): 
    with open('words.txt', 'r', encoding='utf8') as fin, open('words_ok.txt', 'w', encoding='utf8') as fo:
        word_set = set()
        for line in fin:
            word_set.add(line.strip()) if line.strip() else ''
        [fo.write(w+"\n") for w in word_set]
        
def vec_to_word(input):
    with open(input, 'r', encoding='utf8') as fin, open('words_ok.txt', 'w', encoding='utf8') as fo:
        word_set = set()
        for index, line in enumerate(fin):
            segs = line.strip().split()
            if len(segs) != 51:
                print(line)
                continue
            w = segs[0].strip()
            if (w and len(w) == 1) or w in ('<pad>', '<eos>', '<unk>'):
                
                word_set.add(w)
            else:
                print(f"{index} {line} is error")
        [fo.write(w+"\n") for w in word_set]
        
#if __name__ == "__main__":
    #vec_to_word('data/scope/glo_news_50d_ch.txt')
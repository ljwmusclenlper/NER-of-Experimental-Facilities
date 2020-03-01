import sys 
import codecs
def process(emb, vocab, glove): 
    with codecs.open(emb, "r", encoding="utf-8") as emb_f, \
         codecs.open(vocab, "r", encoding="utf-8") as vocab_f, \
         codecs.open(glove, "a", encoding="utf-8") as glove_f:
    
        vocab = set([word.strip() for word in vocab_f])
        done_set = set()
        for line in emb_f:
            try:
                vector = line.strip().split()
                if len(vector) != 301:
                    print(f"{vector[0]} is not 300 dim")
                    continue
                else:
                    word = line.split()[0]
            except:
                continue
            if (word in vocab) and (not (word in done_set)):
                done_set.add(word)
                glove_f.write(line)
            
            
if __name__ == "__main__":
    #$data_dir/vocab.w $data_dir/vocab.t $data_dir/vocab.c \
    process('F:/lager_data/trained_vector/glove.840B.300d/glove.840B.300d.txt','../../data/conll03/vocab.w', '../../data/conll03/eng.glove')

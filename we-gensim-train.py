import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from gensim.models import Phrases

from gensim.models.phrases import Phraser


test_comments = pd.read_csv("gensim.csv", header=None)
test_comments.columns=['comment_id','status_id', 'parent_id', 'comment_message','comment_published', 'num_reactions','num_likes', 'num_loves', 'num_sads', 'num_angrys', 'source','subjectif']
test_comments['comment_message'].replace('', np.nan, inplace=True)
test_comments.dropna(subset=['comment_message'], inplace=True)
comments_text = list(test_comments.comment_message)

comments = []
for comment in comments_text:
    comments.append(word_tokenize(comment))
#df=pd.DataFrame(list(collection.find()))

# define training data

# train model
model = Word2Vec(comments, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['maison'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)
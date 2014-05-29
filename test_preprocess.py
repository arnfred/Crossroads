import sqlite3

from recommender.arxiv.preprocess import build_voc



db_path = 'recommender/data/arxiv.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

vocabulary_filename = 'new_voc.txt'
min_tc = 0
min_df = 0.0
max_df = 1.0

td_sparsemat, vectorizer, voc = build_voc(min_tc, min_df, max_df, cursor)

f = open(vocabulary_filename, 'w')
for word in voc:
    f.write(word+u"\n")
f.close()

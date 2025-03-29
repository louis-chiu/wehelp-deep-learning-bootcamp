from gensim.models import Doc2Vec

PATH = "./output/corpus.cor"
model = Doc2Vec(vector_size=64, min_count=2, epochs=3, workers=12)

model.build_vocab(corpus_file=PATH)
model.train(
    corpus_file=PATH,
    epochs=model.epochs,
)

print(model)

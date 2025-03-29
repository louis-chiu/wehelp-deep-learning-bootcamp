from gensim.models.doc2vec import TaggedDocument


def read_as_tagged_documents(path="./example-data.csv", tokens_only=False):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.split(",")
            if len(parts) > 1:
                if tokens_only:
                    yield parts[1:]
                else:
                    yield TaggedDocument(words=parts[1:], tags=[parts[0]])

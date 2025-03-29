from gensim.models.doc2vec import Doc2Vec
import multiprocessing
from collections import Counter
import logging
from corpus_utils import read_as_tagged_documents
from datetime import datetime

EXECUTE_AT = datetime.now().strftime("%m%d-%H%M")
BASE_PATH = ""  # "./0327-1503/"
PATH = f"{BASE_PATH}example-data.csv"
DEFAULT_MODEL_CONFIG = {
    "vector_size": 8,
    "min_count": 2,
    "epochs": 100,
    "workers": multiprocessing.cpu_count(),
}

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename=f"{BASE_PATH}embedding-{EXECUTE_AT}.log",
)


def evaluate(model, tagged_documents) -> tuple[float, float]:
    ranks = []
    total_corpus = 0
    for doc in tagged_documents:
        inferred_vector = model.infer_vector(doc.words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

        current_tag = doc.tags[0]
        ranked_tags = [tag for tag, _ in sims]

        ranks.append(ranked_tags.index(current_tag))
        total_corpus += 1

    rank_counter = Counter(ranks)
    top1_count = rank_counter.get(0, 0)
    top1_acc = top1_count / total_corpus * 100
    top2_acc = (top1_count + rank_counter.get(1, 0)) / total_corpus * 100
    logging.info(f"Self-Similarity: {top1_acc:.2f}")
    logging.info(f"Second Self-Similarity: {top2_acc:.2f}")

    return int(top1_acc), int(top2_acc)


def setup_model_configuration(config=DEFAULT_MODEL_CONFIG, path=None) -> Doc2Vec:
    if not path:
        logging.info(f"Model Configuration - {config}")
        model = Doc2Vec(**config)
    else:
        logging.info(f"Model Configuration - read model from {path}")
        model = Doc2Vec.load(path)
    return model


def main():
    model = setup_model_configuration()

    logging.info(f"Reading corpus from {PATH}")
    tagged_documents = list(read_as_tagged_documents(PATH))

    model.build_vocab(tagged_documents)
    model.train(
        tagged_documents,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )

    top1_acc, top2_acc = evaluate(model, tagged_documents)

    model.save(f"{BASE_PATH}{EXECUTE_AT}-{top1_acc}-{top2_acc}.model")


if __name__ == "__main__":
    main()

from gensim.models.doc2vec import Doc2Vec
import multiprocessing
from collections import Counter
import logging
from corpus_utils import spllit_data, to_tagged_documents
from datetime import datetime

EXECUTE_AT = datetime.now().strftime("%m%d-%H%M")
BASE_PATH = "./0327-1503/"
PATH = f"{BASE_PATH}tokenized-title"
DEFAULT_MODEL_CONFIG = {
    "vector_size": 64,
    "min_count": 2,
    "epochs": 100,
    "workers": multiprocessing.cpu_count(),
}
RANDOM_STATE = 42



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


def setup_model_configuration(config=DEFAULT_MODEL_CONFIG, path=None):
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
    train_dataset, test_dataset = spllit_data(PATH)
    train_tagged_documents, test_tagged_documents = (
        list(to_tagged_documents(train_dataset)),
        list(to_tagged_documents(test_dataset)),
    )

    model.build_vocab(train_tagged_documents)
    model.train(
        train_tagged_documents,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )

    top1_acc, top2_acc = evaluate(model, test_tagged_documents)

    model.save(f"{BASE_PATH}{EXECUTE_AT}-{top1_acc}-{top2_acc}.model")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
        filename=f"{BASE_PATH}embedding-{EXECUTE_AT}.log",
    )
    
    main()

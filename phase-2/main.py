from utils import ModelUtils, CorpusUtils
from tokenizer import tokenize
from classify import ClassificationNetwork

import torch
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

from typing import cast

EMBEDDING_MODEL_PATH = "0421-2228-89-96.model"
CLASSIFICATION_MODEL_PATH = "0422-2333-94-98.classify.model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_model = ModelUtils.setup_model_configuration(EMBEDDING_MODEL_PATH)
model_state_dict = torch.load(CLASSIFICATION_MODEL_PATH, map_location=DEVICE)
model = ClassificationNetwork().to(DEVICE)
model.load_state_dict(model_state_dict)
model.eval()

WS_DRIVER = CkipWordSegmenter(model="bert-base", device=DEVICE)
POS_DRIVER = CkipPosTagger(model="bert-base", device=DEVICE)
TAG_MAPPING = {index: tag for index, tag in enumerate(embedding_model.dv.index_to_key)}
BOARD_NAMES = [
    "baseball",
    "Boy-Girl",
    "c_chat",
    "hatepolitics",
    "Lifeismoney",
    "Military",
    "pc_shopping",
    "stock",
    "Tech_Job",
]

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/api/model/prediction")
async def get_classification(title: str):
    ws = WS_DRIVER([title])
    ws = WS_DRIVER([title])
    pos = POS_DRIVER(ws)
    tokenized_title = tokenize(ws[0], pos[0], False)
    vectorized_title = CorpusUtils.vectorize(
        cast(list[str], tokenized_title), embedding_model
    ).to(DEVICE)

    if vectorized_title.dim() == 1:
        vectorized_title = vectorized_title.unsqueeze(0)

    prediction: torch.Tensor = model(vectorized_title)
    prediction_label = TAG_MAPPING.get(prediction.argmax().item())
    return {"prediction": prediction_label}


@app.post("/api/model/feedback")
async def submit_feedback(title: str, label: str):
    with open("user-labeled-titles.csv", "a") as file:
        file.write(f"{label}, {title}\n")

    return {
        "message": f"感謝您的回饋！\n已接收到反饋: 標題 '{title}' 應該歸類為 '{label}'"
    }


@app.get("/api/model/boards")
async def get_boards():
    return [board for board in BOARD_NAMES]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

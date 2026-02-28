# genrec/utils/factory.py

from genrec.utils.models_setup.tiger_setup import create_tiger_model
from genrec.utils.models_setup.letter_setup import create_letter_model

from genrec.data.datasets.generative.tiger_dataset import TigerDataset
from genrec.data.collators.generative.tiger_collator import TigerDataCollator

from genrec.quantization.pipelines.rqvae_pipeline import RQVAETrainingPipeline              # 🔥 新增
from genrec.quantization.pipelines.rqvae_pipeline_letter import LETTERRQVAETrainingPipeline

MODEL_FACTORY = {
    "tiger": create_tiger_model,
    "letter": create_letter_model,
}

DATASET_MAP = {
    "tiger": TigerDataset,
    "letter": TigerDataset,
}

COLLATOR_MAP = {
    "tiger": TigerDataCollator,
    "letter": TigerDataCollator,
}

PIPELINE_MAP = {
    "tiger": RQVAETrainingPipeline,
    "letter": LETTERRQVAETrainingPipeline,
}
def get_model_factory(name: str):
    if name not in MODEL_FACTORY:
        raise ValueError(f"Unknown generative type: '{name}', options: {list(MODEL_FACTORY.keys())}")
    return MODEL_FACTORY[name]


def get_dataset_class(name: str):
    if name not in DATASET_MAP:
        raise ValueError(f"Unknown generative type: '{name}', options: {list(DATASET_MAP.keys())}")
    return DATASET_MAP[name]


def get_collator_class(name: str):
    if name not in COLLATOR_MAP:
        raise ValueError(f"Unknown generative type: '{name}', options: {list(COLLATOR_MAP.keys())}")
    return COLLATOR_MAP[name]

def get_pipeline_class(name: str):
    if name not in PIPELINE_MAP:
        raise ValueError(f"Unknown generative type: '{name}', options: {list(PIPELINE_MAP.keys())}")
    return PIPELINE_MAP[name]
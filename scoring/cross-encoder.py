from sentence_transformers import CrossEncoder
from transformers import modeling_utils
    
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import (
    StandardPythonParameterType,
)
import os
import numpy as np
import logging
import json
import pathlib
import time



def init():
    model_dir = os.getenv("AZUREML_MODEL_DIR", "")
    # model_dir = os.path.join(model_dir, "")
    logging.info(f"Model dir: {model_dir}")
    rd = pathlib.Path(f"{model_dir}")
    
    logging.info(f"Contents of model dir {os.listdir(rd)}")
    logging.info(f"Contents of model dir lvl 0 {os.listdir(rd.parents[0])}")
    logging.info(f"Contents of model dir lvl 1 {os.listdir(rd.parents[1])}")
    logging.info(f"Contents of model dir lvl 2 / rubenal {os.listdir(rd.parents[2]/'rubenal')}")
    logging.info(f"Contents of model dir lvl 2 / rubenal / mmarco-mMiniLMv2-L12-H384 {os.listdir(rd.parents[2]/'rubenal'/'mmarco-mMiniLMv2-L12-H384-v1')}")
    model_name = "RubenAMtz/mmarco-mMiniLMv2-L12-H384-v1"
    global model
    model = CrossEncoder(f"{rd.parents[2]}/rubenal/mmarco-mMiniLMv2-L12-H384-v1/", max_length=512, num_labels=1, device='cpu', automodel_args={"local_files_only":True}, tokenizer_args={"use_fast": False})
    logging.info("Model loaded")


@input_schema(
    param_name="data", param_type=StandardPythonParameterType(
        {
            "inputs": {
                "source_sentence": "puedo manejar borracho?", 
                "sentences": ["manejar borracho", "manejar sobrio"]
            }
        })
)
@output_schema(output_type=StandardPythonParameterType( [7.2, 3.2, 6.0, 1.8] ))

def run(data):

    logging.info(type(data))
    logging.info(data)
    # create a list of tuples, where each tuple is "source sentence" and "sentence"
    # the source sentence is repeated for each sentence in the list
    # this is the format that the cross-encoder expects
    start = time.time()
    sentence_pairs = [(data["inputs"]["source_sentence"], sentence) for sentence in data["inputs"]["sentences"]]
    scores = model.predict(sentence_pairs).tolist()
    end = time.time()
    logging.info(f"Time elapsed: {end - start}")
    logging.info(f"Output scores: {scores}")
    logging.info(f"Scores dtype {type(scores)}")
    return scores


# loading the model through the parent folder of the model
# model_path env variable works when there isn't more files that the framework needs for it to load the model. Huggingface requires configuration files on top of the model file.
# to get access to those files, pass the whole 'project' in the deployment stage
# apparently there is a concept of fast tokenizer, a fast tokenizer is a rust implementation of a tokenizer object, this model does not support fast tokenizers, so we need to pass a tokenizer argument "user_fast"=False
# for input_schema, whatever the parameter name is, that is the name of the key in the json that is passed to the run function, for example:
# input_schema(param_name="data", param_type=StandardPythonParameterType({"inputs": {"source_sentence": "puedo manejar borracho?", "sentences": ["manejar borracho", "manejar sobrio"]}}))
# the key in the json is "data", so the run function will receive a json with a key "data" and the value will be the json that is passed to the input_schema function
# {"data": {"inputs": {"source_sentence": "puedo manejar borracho?", "sentences": ["manejar borracho", "manejar sobrio"]}}}
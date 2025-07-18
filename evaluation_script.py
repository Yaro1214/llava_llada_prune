#from model import llada_v
from utils import set_seed
import os
from lmms_eval.__main__ import cli_evaluate
os.environ["LMMS_EVAL_PLUGINS"] = "lmms_plugin"
if __name__ == "__main__":
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    set_seed(1234)
    cli_evaluate()
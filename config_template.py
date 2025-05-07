import os
root_ = os.path.dirname(os.path.abspath(__file__))

# llm api
BASE_URL = ""
API_KEY = ""
MODEL_NAME = "deepseek-chat"

# embedding model
EMBEDDING_MODEL = os.path.join(root_, "bge-large-en-v1.5")
EMBEDDING_CACHE_DIR = os.path.join(root_, "cache")

# data paths
RAW_DATA_ROOT_DIR = os.path.join(root_, "data")
TAG_ROOT_DIR = os.path.join(root_, "tag_data")
PROMPT_1_DIR = os.path.join(root_, "prompt_1")
PROMPT_2_DIR = os.path.join(root_, "prompt_2")

# log paths
LOG_ROOT_DIR = os.path.join(root_, "log")
CMD_LOG_DIR = os.path.join(LOG_ROOT_DIR, "cmd")
PROMPT_LEN_LOG_DIR = os.path.join(LOG_ROOT_DIR, "prompt_len")
FIG_LOG_DIR = os.path.join(LOG_ROOT_DIR, "fig")

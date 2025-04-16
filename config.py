import os
root_ = os.path.dirname(os.path.abspath(__file__))

# llm api
BASE_URL = "http://111.186.56.172:3000/v1"
API_KEY = "sk-e9ZHXSwNdl1gz43v135aCa39A8A943FeBeA059703a6dAfE5"
MODEL_NAME = "deepseek-chat"

# embedding model
EMBEDDING_MODEL = os.path.join(root_, "bge-large-en-v1.5")

# data path
RAW_DATA_ROOT_DIR = os.path.join(root_, "data")
TAG_ROOT_DIR = os.path.join(root_, "tag_data")
PROMPT_1_DIR = os.path.join(root_, "prompt_1")
PROMPT_2_DIR = os.path.join(root_, "prompt_2")

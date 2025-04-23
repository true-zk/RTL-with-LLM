# pip install openai
from openai import OpenAI

api_key = "sk-"
api_key = "sk-"
url = ""


client = OpenAI(
        base_url=url,
        api_key=api_key
    )


def get_models():
    models = client.models.list()
    for model in models:
        print(model.id)


get_models()

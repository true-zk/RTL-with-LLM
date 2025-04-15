# pip install openai
from openai import OpenAI

api_key = "sk-iQn5lQPRtgZRKu56D787847152A34d8aB14dEa157e79FeBf"
api_key = "sk-e9ZHXSwNdl1gz43v135aCa39A8A943FeBeA059703a6dAfE5"
url = "http://111.186.56.172:3000/v1"


client = OpenAI(
        base_url=url,
        api_key=api_key
    )


def get_models():
    models = client.models.list()
    for model in models:
        print(model.id)


get_models()

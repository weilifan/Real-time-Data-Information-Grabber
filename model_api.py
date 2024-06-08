from flask import request, Flask
from transformers import AutoTokenizer, AutoModel
import os

app = Flask(__name__)


@app.route("/answer", methods=["POST"])
def send():
    global model, tokenizer

    param = request.json

    query = param["query"]
    system_info = eval(param["system_info"])

    res, his = model.chat(tokenizer, query, history=[system_info])

    result = {
        "ans": res
    }

    return result


if __name__ == "__main__":
    model_path = os.path.abspath("D:/Project/2309/Real-time-Data-Information-Grabber/chatglm3-6b-chat")

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto", offload_folder="offload_folder").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = model.eval()

    # app.run(host="0.0.0.0",port=8080)
    app.run(host="127.0.0.1")
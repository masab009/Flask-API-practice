from flask import Flask,request,jsonify
import torch
import numpy as np
from transformers import RobertaTokenizer,RobertaForSequenceClassification
import onnxruntime
import os
app = Flask("Roberta sentiment analysis")
current_directory = os.getcwd()
tokenizer = RobertaTokenizer.from_pretrained("roberta-base",cache_dir=current_directory)
model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment",cache_dir=current_directory)

def to_numpy(tensor):
    return (
        tensor.detach().cpu.numpy() if tensor.requires_grad else tensor.cpu().numpy()
        )

@app.route("/")
def home():
    return "<h2>roBERTA sentiment analysis</h2>"
      
@app.route("/predict",methods=["POST"])
def getPredictions():
    data = request.json
    input_text = data["text"]
    if not input_text:
        return jsonify({"error": "Text field is required"}), 400  
    inputs = tokenizer.encode_plus(
        input_text,
        return_tensors = "pt",
        truncation = True,
        padding = True,
        max_length = 512
    )

    output = model(**inputs)
    logits = output.logits
    probabilities = torch.softmax(logits,dim=-1)    
    probabilities = probabilities.tolist()[0]
    
    response = {
        "Negative": probabilities[0],
        "Neutral" : probabilities[1],
        "Positive" : probabilities[2]
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
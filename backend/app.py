from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

model = GPT2LMHeadModel.from_pretrained("path/to/save/your-model")
tokenizer = GPT2Tokenizer.from_pretrained("path/to/save/your-model")

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get("prompt", "")
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)

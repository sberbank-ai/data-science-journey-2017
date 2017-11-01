from flask import Flask, request, jsonify

app = Flask(__name__)


# import your predict functions here
# load your models here, for instance:
#
# from my_solution import SDSDPredictor
# model = SDSJPredictor('./my_model.pickle')
#
# or for simple-baseline:
from predict import get_max_match_sentance


def make_answer(paragraph, question):
    # insert here your prediction code
    # e.g.:
    return get_max_match_sentance({'paragraph': paragraph, 'question': question})


@app.route('/predict', methods=['POST'])
def predict_handler():
    data = request.get_json()
    answer = make_answer(data['paragraph'], data['question'])
    return jsonify({'answer': answer})


@app.route('/health')
def health_handler():
    return jsonify({'ok': True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

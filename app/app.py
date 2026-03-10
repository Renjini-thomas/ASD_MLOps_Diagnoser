from flask import Flask, render_template, request
import os

from src.predict.inference_pipeline import InferencePipeline
import uuid
app = Flask(__name__)

pipeline = InferencePipeline()

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])

def index():

    prediction = None
    probability = None

    if request.method == "POST":

        file = request.files["file"]
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        prediction, prob = pipeline.predict(filepath)

        probability = prob.max()

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )


if __name__ == "__main__":
    app.run(debug=True)
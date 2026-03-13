from flask import Flask, render_template, request
import os
import uuid

from src.predict.inference_pipeline import ASDInference

app = Flask(__name__)

pipeline = ASDInference()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    probability = None
    error = None

    if request.method == "POST":

        if "file" not in request.files:
            error = "No file uploaded"
            return render_template(
                "index.html",
                prediction=prediction,
                probability=probability,
                error=error
            )

        file = request.files["file"]

        if file.filename == "":
            error = "Empty file selected"
            return render_template(
                "index.html",
                prediction=prediction,
                probability=probability,
                error=error
            )

        try:

            filename = str(uuid.uuid4()) + "_" + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            file.save(filepath)

            prediction, probability = pipeline.predict(filepath)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
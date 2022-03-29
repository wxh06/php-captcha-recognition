from flask import Flask, flash, redirect, render_template, request
from predict import predict as _predict

ALLOWED_EXTENSIONS = {"jpg", "jpeg"}

app = Flask(__name__)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def upload():
    return render_template("upload.html")


@app.route("/predict/", methods=["POST"])
def predict():
    # check if the post request has the file part
    if "captcha" not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files["captcha"]
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        return _predict(file.stream)
    return ("", 400)

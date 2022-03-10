from flask import Flask, render_template, request
from werkzeug.utils import format_string, redirect, secure_filename
import os
import cv2

from Pytorch_opencv import predictImg, ret_prediction
app = Flask(__name__)

path_for_store = os.path.join(os.path.dirname(os.path.realpath(__file__)),'upload')

app.config["IMAGE_UPLOADS"] = path_for_store

app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF",'MP4']

opencv_net_c40 = cv2.dnn.readNetFromONNX('raw_93.onnx')
opencv_net_raw = cv2.dnn.readNetFromONNX('raw_93.onnx')


def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


@app.route('/')
def form():
    return render_template('public/faceforensics.html')

@app.route("/upload-file", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]

            if image.filename == "":
                print("No filename")
                return redirect(request.url)

            if allowed_image(image.filename):
                filename = secure_filename(image.filename)

                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                print("Image saved")
                try:
                    model_option = int(request.form.getlist('options')[0])
                    type_option = int(request.form.getlist('types')[0])
                    if model_option==1:
                        value=ret_prediction(path=os.path.join(app.config["IMAGE_UPLOADS"], filename),type_option=type_option,opencv_net=opencv_net_raw)
                    else:
                        value=ret_prediction(path=os.path.join(app.config["IMAGE_UPLOADS"], filename),type_option=type_option,opencv_net=opencv_net_c40)

                    print(model_option)
                    print(type_option)

                    if value:
                        if type_option==11:
                            prediction = 'The image is Original'
                        else:
                            prediction = 'The Video is Original'
                    else:
                        if type_option==11:
                            prediction = 'The image is Fake'
                        else:
                            prediction = 'The Video is Fake'

                    #----------------------------------------------

                    file_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
                    os.remove(file_path)

                    return render_template("public/faceforensics.html",value = prediction)

                    #----------------------------------------------
                except:
                    print('not selected')
                return redirect(request.url)

            else:
                print("That file extension is not allowed")
                return redirect(request.url)
    


if __name__ == '__main__':
    app.run(debug=True)

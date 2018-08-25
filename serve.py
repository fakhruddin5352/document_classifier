# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import load_model as load_keras_model
import tensorflow as tf

from PIL import Image
import numpy as np
import flask
from flask_cors import CORS as cors
import io

included = ['Birth Certificate',
'EIDA Card',
'Employment Contract',
'Entry permit visa - Lost letter from police',
'House Rental contract',
'Passport',
'Residency',
'Salary Certificate-Labour contract-Partnership Contract',
'Sponsored Photo'
]


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
cors(app)

modelFile = 'models/vgg19.h5'
model = None
graph = None
def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global graph
    graph = tf.get_default_graph()
    global model
    print(f'Loading model {modelFile}')
    model = load_keras_model(modelFile)

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target, Image.ANTIALIAS)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    # return the processed image
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False,"predictions":{}}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        for name in flask.request.files:
            # read the image in PIL format
            image = flask.request.files[name].read()
            
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(299, 299))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            with graph.as_default():
                preds = model.predict(image)
            data["predictions"][name] = []

            # loop over the results and add them to the list of
            # returned predictions
            for i,c in enumerate(included):
                r = {"label": c, "probability":float(preds[0][i])}
                data["predictions"][name].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
load_model()

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(port=8080)  
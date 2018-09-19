from flask import Flask
from flask import jsonify
from flask_cors import CORS
from flask import request
from skimage import feature
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import requests

app = Flask(__name__)
cors = CORS(app)
    
input_shape = (136,)
left_input = Input(input_shape, name = 'left')
right_input = Input(input_shape, name = 'right')
#build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(Dense(500,activation="relu"))

#call the convnet Sequential model on each of the input tensors so params will be shared
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
#layer to merge two encoded inputs with the l1 distance between them
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
#call this layer on list of two input tensors.
L1_distance = L1_layer([encoded_l, encoded_r])
prediction = Dense(1,activation='relu')(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam()
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
siamese_net.load_weights('siamese_conv_trained.h5')

from imutils import face_utils
import imutils
import dlib
import cv2

def getFace_Cords(img_name):
    if(isinstance(img_name, str)):
        image = cv2.imread(img_name)
    else:
        image = img_name
        
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    rects = detector(gray, 1)
    shape = predictor(gray, rects[0])
    shape = face_utils.shape_to_np(shape)
    return shape

def computeAbsDifference(coords1, coords2):
    metrics = []
    if(coords1.shape != coords2.shape):
        raise Exception('Coords need to be the same shape')
    for i in range(0,2):
        diff = np.abs(coords1[:, i] - coords2[:, i])
        metrics.append(np.mean(diff))
    return metrics

def getPrediction(net, img1, img2):
    coord1 = getFace_Cords(img1)
    coord2 = getFace_Cords(img2)
    coords1 = coord1 - np.mean(coord1)
    coords2 = coord2 - np.mean(coord2)
    coord1 = coords1.flatten()
    coord2 = coords1.flatten()
    x = siamese_net.predict({'left': coord1.reshape(1, 136),
                                'right': coord2.reshape(1, 136)
                                })
    #return sum(computeAbsDifference(coords1, coords2))
    return x

def convertback(img):
    image = Image.fromarray(img)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue())
        return img_str.decode('utf-8')

@app.route('/')
def hello_world():
    print('We Were just toggled')
    return 'Hello from Sachin Flask!'

@app.route('/old', methods = ['POST'])
def getbw():
    data = request.get_json()
    print(data.keys())
    person = data['person']
    image = data['image']
    string = data['imgdata'].split(',')
    header = string[0]
    converted = BytesIO(base64.b64decode(string[1]))
    img1 = np.array(Image.open(converted))
    predicted = getPrediction(siamese_net, img1, 'P%s/img%d.jpg'%(person, image) )
    print(predicted)
    return jsonify({'predicted': float(predicted[0][0])})


if __name__ == '__main__':
  app.run(host="0.0.0.0")
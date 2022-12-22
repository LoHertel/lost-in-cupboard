import os
from io import BytesIO
from urllib import request as rq

import grpc
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

MODEL_NAME = 'kitchenware-model'
INPUT_NAME = 'input_4'
OUTPUT_NAME = 'dense_3'
IMAGE_SIZE = (150, 150)
CLASSES = ['cup', 'fork', 'glass', 'knife', 'plate', 'spoon']


app = Flask('prediction-gateway')

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')
channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


def dtypes_as_dtype(dtype):
    if dtype == 'float32':
        return types_pb2.DT_FLOAT
    raise Exception(f'dtype {dtype:s} is not supported')


def make_tensor_proto(data):
    shape = data.shape
    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=i) for i in shape]
    proto_shape = tensor_shape_pb2.TensorShapeProto(dim=dims)

    proto_dtype = dtypes_as_dtype(data.dtype)

    tensor_proto = tensor_pb2.TensorProto(dtype=proto_dtype, tensor_shape=proto_shape)
    tensor_proto.tensor_content = data.tobytes()

    return tensor_proto


def np_to_protobuf(data):
    if data.dtype != 'float32':
        data = data.astype('float32')
    return make_tensor_proto(data)


# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def preprocess_from_url(urls: list[str]):

    X = []

    if type(urls) != list:
        urls = [urls]

    for url in urls:

        # load image from url
        with rq.urlopen(url) as resp:
            buffer = resp.read()
        stream = BytesIO(buffer)
        img = Image.open(stream)
        
        # convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # resize image
        img = img.resize(IMAGE_SIZE, Image.Resampling.NEAREST)

        X.append(np.array(img, dtype='float32'))
    
    # turn to 4d numpy array (1, img_size, img_size, 3)
    X = np.array(X)

    # convert to protobuf
    return np_to_protobuf(X)


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = MODEL_NAME
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs[INPUT_NAME].CopyFrom(X)
    return pb_request


def prepare_response(pb_response):
    float_predictions = pb_response.outputs[OUTPUT_NAME].float_val

    result = []

    for float_prediction in divide_chunks(float_predictions, len(CLASSES)):
        result.append(dict(zip(CLASSES, float_prediction)))

    return result



def predict(urls):

    if type(urls) != list:
        urls = [urls]

    X = preprocess_from_url(urls)
    pb_request = prepare_request(X)
    result = prepare_response(stub.Predict(pb_request, timeout=20.0))

    if len(result) == 1:
        return result[0]

    return result


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    urls = request.get_json()

    if type(urls) != list:
        urls = [urls]

    result = predict(urls)
    return jsonify(result)


if __name__ == '__main__':
    # url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Mocha_cup%2C_designed_by_Adolf_Flad%2C_made_by_KPM_Berlin%2C_1902%2C_porcelain%2C_1_of_6_-_Br%C3%B6han_Museum%2C_Berlin_-_DSC04094.JPG/640px-Mocha_cup%2C_designed_by_Adolf_Flad%2C_made_by_KPM_Berlin%2C_1902%2C_porcelain%2C_1_of_6_-_Br%C3%B6han_Museum%2C_Berlin_-_DSC04094.JPG'
        
    # result = predict(url)
    # print(result)
    app.run(debug=True, host='0.0.0.0', port=9696)
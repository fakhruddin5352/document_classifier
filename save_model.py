import keras.backend as K
import tensorflow as tf
from keras.models import load_model, Sequential
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

# reset session
K.clear_session()
sess = tf.Session()
K.set_session(sess)

# disable loading of learning nodes
K.set_learning_phase(0)


# load model
model = load_model('snapshot/vgg19.final.2018-08-21 19:36.h5')
config = model.get_config()
weights = model.get_weights()
new_Model = model
new_Model = Sequential.from_config(config)
new_Model.set_weights(weights)

# e`x`port saved model
export_path = 'models/export1'
builder = saved_model_builder.SavedModelBuilder(export_path)

signature = predict_signature_def(inputs={'X': new_Model.input},
                                  outputs={'y': new_Model.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={
                                             signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
    builder.save()

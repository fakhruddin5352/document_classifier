import numpy as np
import os
import csv
from pathlib import Path
import urllib.request
from PIL import Image
from time import time
import json

import keras
from keras import Model
from keras.applications import VGG19
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Flatten,Dropout
from keras.utils import np_utils
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

now = datetime.now().strftime("%Y-%m-%d %H:%M")

class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'snapshot/weights_2cat_1.h5'
            self.model.save_weights(name)
        self.batch += 1


def map_document_type(type, name):
    if type in ['00000000000000000000000000000115','00000000000000000000000000000113','00000000000000000000000000000116','00000000000000000000000000000114','00000000000000000000000000000117','00000000000000000000000000000123']:
        return 'Three Months Bank Statement'
    if type in ['470F296EE30445E08C023BE290362C7B','595EB63E2AE94544BD28DD27F81C6E75','5E424ABF04A545C3B42A9D28602245C4','00000000000000000000000000000062','00000000000000000000000000000096','00000000000000000000000000000060','00000000000000000000000000000078','00000000000000000000000000000095','00000000000000000000000000000041','00000000000000000000000000000070','00000000000000000000000000000072',
                '00000000000000000000000000000075','EF3DB9C0B60C4F1BB072DFC640903DE6','1FF6835C8A964C849183007D3E9E69D7','00000000000000000000000000000097','00000000000000000000000000000077','00000000000000000000000000000093','00000000000000000000000000000163','00000000000000000000000000000061','00000000000000000000000000000094','00000000000000000000000000000098','00000000000000000000000000000101',
                '00000000000000000000000000000099','00000000000000000000000000000073']:
        return 'Passport'
    if type in ['94AF83C5A7BD45D987392B7EE410210F','00000000000000000000000000000102','E5171A18D2254C8B9942848AC31DC1F9']:
        return 'Residency'
    if type in ['00000000000000000000000000000107','00000000000000000000000000000066','00000000000000000000000000000065','00000000000000000000000000000067']:
        return 'Salary Certificate-Labour contract-Partnership Contract'
    if type in ['CDD3E0173F2641D9A82B472455604E73','00000000000000000000000000000055','00000000000000000000000000000103']:
        return 'Salary Certificate-Labour contract-Partnership Contract'
    if type in ['ACC17E81760B44BD8EFA439F8CD8DAD1','BAADA8086BC54C67BADF4E215BA61194','00000000000000000000000000000146','00000000000000000000000000000145']:
        return 'House Rental contract'
    if type in ['00000000000000000000000000000079','00000000000000000000000000000129','00000000000000000000000000000082','00000000000000000000000000000083','00000000000000000000000000000084','00000000000000000000000000000085',
                '07C872F21DBA4C4CB666ED2CE5303DC0','F1668EA64038438F8739F8420E09092A','00000000000000000000000000000087','00000000000000000000000000000086','00000000000000000000000000000131','00000000000000000000000000000081',
                '00000000000000000000000000000130','00000000000000000000000000000133','00000000000000000000000000000132','00000000000000000000000000000080','00000000000000000000000000000128']:
        return 'Employment Contract'
    if type in ['00000000000000000000000000000160','00000000000000000000000000000162','00000000000000000000000000000161','00000000000000000000000000000154']:
        return 'Medical Report'
    if type in ['00000000000000000000000000000037','025E6D51961E4204AFA40BEC20912E3E','00000000000000000000000000000180']:
        return 'Birth Certificate'
    if type in ['EF958400A9F141A996FC8B7F4587752D','00000000000000000000000000000190']:
        return 'Entry permit visa - Lost letter from police'
    if type in ['00000000000000000000000000000068','5C608DC460534447AEF54CCDD492DD0D']:
        return 'EIDA Card'

    if type in ['EBCAE64ACB1441308FC6B7543182BEEB','785577AF6E8A4F369C51CF3A079DB8AE','1ADED610F4434C638CD8FD2BD505D50A','FA5702B7ED474484BC7F051211B13229',
                '00000000000000000000000000000177','5046883DE440423A87908255EE0520A0','66AB514E1FBC4C91B7FE6CE01FAC29B7','00000000000000000000000000000178',
                '00000000000000000000000000000176','4A00A65FDE3447828979B41DAAAD03EA','6154B1AB47734C4E9782186F9C7A2BC4','00000000000000000000000000000179',
                'F880139677C84DDDB4BEF85C72A56D70','C17351F994104B35B35E7463CA8B099D','48E5537414554BC48B4E54879030466D','A87580D23FD24E0A985742215C9EB5A3',
                '000000000F0000000000000000000111','00000000000000000000000000000175','0F13C6FEDC46462897153D4CA53C9F30','000000000F0000000000000000000112']:
        return 'Uncategorized'
    if type in ['00000000000000000000000000000053','00000000000000000000000000000038','00000000000000000000000000000151','F29169DDC5C944188285242739570BEB','18DE97A9DBED45F7B101E989B146FF90']:
        return 'Uncategorized'
    if type in ['00000000000000000000000000000148','00000000000000000000000000000149']:
        return 'Uncategorized'
    if type in ['00000000000000000000000000000030','00000000000000000000000000000058']:
        return 'Uncategorized'
    if type in ['00000000000000000000000000000047','82B8292BCED94835A91364FD87A3FD52','00000000000000000000000000000033','00000000000000000000000000000035']:
        return 'Uncategorized'
    if type in ['00000000000000000000000000000005','00000000000000000000000000000027']:
        return 'Uncategorized'
    if type in ['00000000000000000000000000000017','46C0C140F0E34732A123B47F45986261']:
        return 'Uncategorized'
    if type in ['77DDF4E7CF054FCCB7B994057069D701','00000000000000000000000000000183']:
        return 'Uncategorized'

    if type in ['00000000000000000000000000000015','00000000000000000000000000000014','00000000000000000000000000000165','F6310223224F43788A45DBA40FD2E1E9']:
        return 'Uncategorized'
    if type in ['BAE04485D2834A83A517F6DCF14FD775','00000000000000000000000000000191']:
        return 'Uncategorized'
    if type in ['00000000000000000000000000000031','CCA1234B4C034C69947390BA803E122A']:
        return 'Uncategorized'
    if type in ['55D58D962F594AD19768BBEAFD775BA3','7763C57D4D804F409EA7F10735C04349','23CD0CB8DADF449FBF904B6ECD5C63C6','78C5C2F9B23A4B6D88ED806A18246475',
                '00000000000000000000000000000007','00000000000000000000000000000166','00000000000000000000000000000008','00000000000000000000000000000059',
                '1DFE073824124861A0FEEB9E7873C636','08517132F3954DD9B6B59394278F4110','158DB56C997C41F38C1C0FB8C70BAF59','00000000000000000000000000000106',
                '7DF253B7AD5242B29D10F6D661DD9BC8','000000000F0000000000000000000113','00000000000000000000000000000063','00000000000000000000000000000173',
                '80618C6769A348C2B67EA14D87D8B6BA','3963DC12A2AD4D53919592909C3B37FF','00000000000000000000000000000091','00000000000000000000000000000142',
                'AC65FDC3F3A6451B9A81D411A31D0649','00000000000000000000000000000143','00000000000000000000000000000186','00000000000000000000000000000193',
                '00000000000000000000000000000009','9C794D7F36C04162B7D5C8190B800B0F','00000000000000000000000000000064','8877C6B536C447B99286EE012956D672',
                '00000000000000000000000000000040','00000000000000000000000000000181','00000000000000000000000000000147','543439B8221045409172EAA7B12B53AF',
                '9A7D2FADE57C4EFFB0D129DAE57CA7D0','A420137CFD0D463F88AC105642588748']:
        return 'Uncategorized'
    return name


class DocumentType:
    def __init__(self, name, label):
        self.label = label
        self.name = name


class Document:
    def __init__(self, id, type):
        self.id = id
        self.type = type


def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:            
            memo[x] = f(x)
        return memo[x]
    return helper

def load_document_by_id(path):
    img = Image.open(path)
    # arr = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
    arr1 = np.asarray(img)
    #arr1 = self.__normalized(arr1)
    

    s = np.shape(arr1)
    if len(s) == 3 and s[2] == 3:
        return arr1
    else:
        raise 'err'

class DataGenerator(keras.utils.Sequence):
    def __init__(self, name,documents,n_classes, batch_size=32,dim=(299,299), n_channels=1,
                  shuffle=True ):
        'Initialization'
        self.batch_size = batch_size
        self.documents = documents
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.dim = dim
        self.indexes = []
        self.on_epoch_end()
        self.empty_image = np.zeros((self.dim[0], self.dim[1], self.n_channels))
        self.download_count = 0
        self.discard_count = 0
        self.name = name
        self.cached_load = load_document_by_id

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.documents) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        document_temp = [self.documents[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(document_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.documents))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __normalized(self, rgb):
        rgb  =rgb/1.
        sum = rgb[:,:,0]+rgb[:,:,1]+rgb[:,:,2]+0.001

        rgb[:,:,0] = rgb[:,:,0]/sum
        rgb[:,:,1] = rgb[:,:,1]/sum
        rgb[:,:,2] = rgb[:,:,2]/sum
        return rgb

    

    def __load_document(self, document):
        document_type = document.type.name
        document_id = document.id
        document_dir = f"../documents/{document_type}"
        arr = self.empty_image
        try:
            document_conv_path = f"{document_dir}/{document_id}_{self.dim[0]}_{self.dim[1]}.jpg"
            arr = self.cached_load(document_conv_path)
        except Exception as e:
            print (str(e))
            self.discard_count = self.discard_count + 1
        #    print('\n' + self.name + ' Discarded ' + str(self.discard_count))
        '''if self.discard_count > 0 and self.discard_count % 5 == 0:
            '''

        return arr

    def __data_generation(self, document_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, document in enumerate(document_temp):
            # Store sample
            X[i,] = self.__load_document(document)
            # Store class
            y[i] = document.type.label

        c = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, c


def read_csv(file_name):
    rows = []
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        index = 0
        for row in reader:
            if index > 0:
                rows.append(row)
            index = index + 1
    return rows


def read_document_types(file_name,included):
    rows = read_csv(file_name)
    dts = []
    unmapped = {}
    index = 0
    for row in rows:
        type_id = row[0]
        name = map_document_type(row[0],row[1])
        unmapped[type_id] = name
        if len([w for w in dts if w.name == name]) == 0:
            dts.append(DocumentType(name, index))
            index = index+1

    mapped = dict((d.name,d) for d in dts if d.name in included)
    return mapped


def try_read_document(document_id, document_types):
    for document_type_name in document_types:
        document_dir = f"../documents/{document_type_name}"
        my_dir = Path(document_dir)
        if not my_dir.is_dir():
            my_dir.mkdir()

        document_path = f"{document_dir}/{document_id}_{Image_width}_{Image_height}.jpg"
        my_file = Path(document_path)
        if my_file.is_file():
            return Document(document_id, document_types[document_type_name])
    return None
    
def read_documents(file_name, document_types):
    rows = read_csv(file_name)
    documents = [try_read_document(w[0], document_types) for w in rows ]
    documents = [x for x in documents if x is not None]
    return documents


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
document_types = read_document_types('document_type.csv',included)
for i,dt in enumerate(included):
    document_types[dt].label = i


print('Number of categories: ' + str(len(included)))
print(json.dumps(document_types,default=lambda obj: vars(obj)))


Image_width, Image_height = 299, 299
Training_Epochs = 40
Batch_Size = 32
Number_FC_Neurons = 1024


# Define data pre-processing
#   Define image generators for training and testing
documents = [x for x in read_documents('document.csv',document_types) if x.type.name  in included]

batch_size = Batch_Size
num_train_samples = ((len(documents)*70) // (100*batch_size)) * batch_size
num_classes = len(included)
num_validate_samples = ((len(documents)*10) // (100*batch_size)) * batch_size 
num_epoch = Training_Epochs

print("Total training size = ", num_train_samples, ' validate size = ', num_validate_samples)
for document_type in document_types:
    print (document_type + ' ' + str(len([x for x in documents[num_validate_samples:num_train_samples+num_validate_samples] if x.type.name == document_type  ])) + 
   ',' + str(len([x for x in documents[0:num_validate_samples] if x.type.name == document_type  ])))

train_image_gen = DataGenerator('Train', documents[num_validate_samples:num_train_samples+num_validate_samples], num_classes, 
                               batch_size, (Image_width, Image_height), n_channels=3, shuffle=True)
test_image_gen = DataGenerator('Test', documents[0:num_validate_samples], num_classes,
                               batch_size, (Image_width, Image_height), n_channels=3, shuffle=True)
#train_image_gen = generator(0, num_train_samples, 'train')
#test_image_gen = generator(num_train_samples, num_validate_samples,'test')

# Load the Inception V3 model and load it with it's pre-trained weights.  But exclude the final
#    Fully Connected layer

InceptionV3_base_model = VGG19(weights='imagenet', include_top=False, input_shape=(Image_width,Image_height,3)) #include_top=False excludes final FC layer
print('Vgg19 base model without last FC loaded')
#print(InceptionV3_base_model.summary())     # display the Inception V3 model hierarchy

x = InceptionV3_base_model.output

x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(num_classes, activation="softmax")(x)
# Define trainable model which links input from the Inception V3 base model to the new classification prediction layers
model = Model(inputs=InceptionV3_base_model.input, outputs=predictions)


for layer in InceptionV3_base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(now))

# train the model on the new data for a few epochs
'''model.fit_generator(    train_image_gen,
    epochs=num_epoch // 2,
    steps_per_epoch=num_train_samples // batch_size,
    class_weight='auto',
    validation_data=test_image_gen,
    validation_steps=num_validate_samples // batch_size,
    callbacks=[WeightsSaver(model, 200),tensorboard])
'''

# Option 1: Basic Transfer Learning
print ('\nPerforming Transfer Learning')
#   Freeze all layers in the Inception V3 base model

for layer in model.layers[:5]:
   layer.trainable = False
for layer in model.layers[5:]:
   layer.trainable = True

# print model structure diagram
#print(model.summary())

#   Define model compile for basic Transfer Learning
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('snapshot/vgg19.final.2018-08-21 19:36.h5')

# Fit the transfer learning model to the data from the generators.
# By using generators we can ask continue to request sample images and the generators will pull images from
# the training or validation folders and alter them slightly

checkpoint = ModelCheckpoint(f"snapshot/vgg19.{now}.{num_classes}-"+"{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=2, verbose=1, mode='auto')

history_transfer_learning = model.fit_generator(
    train_image_gen,
    epochs=num_epoch,
    steps_per_epoch=num_train_samples // batch_size,
    class_weight='auto', 
    validation_data=test_image_gen,
    validation_steps=num_validate_samples // batch_size,
    callbacks=[checkpoint, early,tensorboard]
) 

# Save transfer learning model
model.save(f'snapshot/vgg19.final.{now}.h5')

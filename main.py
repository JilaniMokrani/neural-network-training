from tensorflow.keras.applications.vgg19 import VGG19

# Creating Network

img_rows, img_cols = 480, 640
nb_labels = 10


model = VGG19( weights= "imagenet",
    include_top = False,
    input_shape = (img_rows,img_cols,3)
    )

def layerAdder(bottomModel, numClasses):
    topModel = bottomModel.output
    topModel = GlobalAveragePooling2D()(topModel)
    topModel = Dense(1024, activation="relu")(topModel)
    topModel = Dense(512, activation="relu")(topModel)
    topModel = Dense(numClasses, activation="softmax")(topModel)
    return topModel

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

FC_Head = layerAdder(model,nb_labels)

model = Model(inputs = model.inputs, outputs = FC_Head)

print(model.summary())

#Importing data

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import environ
from PIL import Image

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

data_dir = environ["DATA_DIR"]
batch_size=10

train_dataset = train_datagen.flow_from_directory(
    data_dir + "/train",
    color_mode="rgb",
    subset="training",
    shuffle =True,
    seed=123,
    batch_size=batch_size,
    target_size= (img_rows,img_cols)
    )


validation_dataset = test_datagen.flow_from_directory(
    data_dir + "/test",
    color_mode="rgb",
    subset="validation",
    shuffle =True,
    seed=123,
    batch_size=batch_size,
    target_size= (img_rows,img_cols)
)

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint(
    "/models/song_classifier",
    monitor= "val_loss",
    mode = "min",
    save_best_only = True,
    verbose = 1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    min_delta= 0,
    patience= 3,
    verbose= 1,
    restore_best_weights= True
)

callbacks = [earlystop, checkpoint]
epochs = 1

model.compile(
    optimizer= 'sgd',
    loss= MeanSquaredError(),
    metrics= ["accuracy"]
)
print("Is starting training! \n")
model.fit(
    train_dataset,
    epochs= epochs,
    verbose= 1,
    callbacks = callbacks,
    validation_data=validation_dataset
)

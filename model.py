#import the necessary modules
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import zipfile
import matplotlib.pyplot as plt
import numpy as np

#unzipping the data
zip_ref = zipfile.ZipFile('garbagedata.zip', 'r')
zip_ref.extractall("/split-garbage-dataset")
zip_ref.close()

#understanding our data
image_count = 0
labels = []
train_counts = []
for dirname in os.listdir('split-garbage-dataset/train'):
    labels.append(dirname)
    image_count = 0
    for img in os.listdir(os.path.join('split-garbage-dataset/train',dirname)):
        image_count +=1
    train_counts.append(image_count)

class_weights = []
total_samples = train_generator.samples
total_classes = len(train_generator.class_indices)
for ele in train_counts:
    result = round(total_samples / (total_classes * ele),2)
    class_weights.append(result)

class_weights = dict(zip(train_generator.class_indices.values(),class_weights))


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                  )

# Note that the validation data should not be augmented!
valid_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 8 using train_datagen generator
train_generator = train_datagen.flow_from_directory('../input/split-garbage-dataset/train',
                                                    batch_size =8,
                                                    class_mode = 'categorical',
                                                    target_size = (150,150))

# Flow validation images in batches of 4 using valid_datagen generator
validation_generator = valid_datagen.flow_from_directory( '../input/split-garbage-dataset/valid',
                                                          batch_size  = 4,
                                                          class_mode  = 'categorical',
                                                          target_size = (150,150))


#Build model

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

xception_model = keras.applications.xception.Xception(weights="imagenet",
                                                      include_top=False,
                                                      input_shape = (150,150,3))
avg = keras.layers.GlobalAveragePooling2D()(xception_model.output)
output = keras.layers.Dense(6, activation="softmax")(avg)
model2 = keras.models.Model(inputs=xception_model.input, outputs=output)

#Compile the model
optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
model2.compile(optimizer =optimizer ,loss = 'categorical_crossentropy',metrics =['accuracy'])

#Define exponential decay learning rate scheduler
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.001, s=10)

#callbacks to be used while training
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 10,restore_best_weights = True)
mc =tf.keras.callbacks.ModelCheckpoint('XceptionLR_Xce2exp.h5', save_best_only=True)

class CustomCallBack(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if(logs.get('accuracy')>0.99):
                print("\nReached 99.0% accuracy so cancelling training!")
                self.model.stop_training = True
mycallback = CustomCallBack()

#train the model
history1 = model2.fit(
            train_generator,
            steps_per_epoch=train_generator.samples/train_generator.batch_size,
            epochs = 30,
            validation_data = validation_generator,
            validation_steps= validation_generator.samples/validation_generator.batch_size,
            callbacks= [early_stopping_cb,mc,lr_scheduler,mycallback],
            verbose=1)

#Understanding the accuracy and loss of the model
fig = plt.figure(figsize=(10,10))

# Plot accuracy
plt.subplot(221)
plt.plot(history1.history['accuracy'],'bo-', label = "acc")
plt.plot(history1.history['val_accuracy'], 'ro-', label = "val_acc")
plt.title("train_accuracy vs val_accuracy")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.grid(True)
plt.legend()

# Plot loss function
plt.subplot(222)
plt.plot(history1.history['loss'],'bo-', label = "loss")
plt.plot(history1.history['val_loss'], 'ro-', label = "val_loss")
plt.title("train_loss vs val_loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.grid(True)
plt.legend()

plt.subplot(223)
plt.plot(history1.epoch,history1.history['lr'],'o-')
plt.title("train_loss vs val_loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.grid(True)
plt.legend()


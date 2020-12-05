import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

#load the saved model
model2 = keras.models.load_model('XceptionLR_Xce2exp.h5')

# Flow test.py images using test_datagen generator
test_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_generator = test_datagen.flow_from_directory('../input/split-garbage-dataset/test.py',
                                                  batch_size = 1,
                                                  class_mode = 'categorical',
                                                  target_size = (150,150),shuffle = False)
#evaluate the model
model2.evaluate(test_generator,batch_size = 1)

#make predictions on test set
y_pred = model2.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(test_generator.classes, y_pred))
cf_matrix = confusion_matrix(test_generator.classes, y_pred)
plt.figure(figsize=(10,10))
heatmap = sns.heatmap(cf_matrix, xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys(), annot=True, fmt='d', color='red')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion matrix of model')

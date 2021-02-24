from flask import Flask, render_template, request, redirect
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask import flash
from flask import session
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
import os
import glob
import shutil

from keras.applications.vgg19 import VGG19

import numpy as np
import sklearn
#import Keras packages
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import load_img
import random
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

import numpy as np
from keras import applications
from keras.layers import Input
from keras.models import Model,load_model
from keras import optimizers
from keras.utils import get_file


src_dir = "/static/img"
dst_dir = "/.static/img1"

app = Flask(__name__)
app.secret_key = "super secret key"

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.route('/', methods=['GET', 'POST'])
def upload():
    flash('')
    if request.method == 'POST' and 'photo' in request.files:
        	filename = photos.save(request.files['photo'])
        	for jpgfile in glob.iglob(os.path.join(src_dir, "*.*")):
        	  shutil.copy(jpgfile, dst_dir)
        
        	vgg = VGG19(weights=None, input_shape=(256,256,3), include_top=False)


        	# don't train existing weights
        	for layer in vgg.layers:
        	  layer.trainable = False
          
          
        
          
        
          
        
        	# our layers - you can add more if you want
        	x = Flatten()(vgg.output)
        	# x = Dense(1000, activation='relu')(x)
        	prediction = Dense(2, activation='softmax')(x)
        
        	# create a model object
        	model = Model(inputs=vgg.input, outputs=prediction)
        	# view the structure of the model
        	#model.summary()
        
        	# tell the model what cost and optimization method to use
        	model.compile(
          loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy']
        	)
        
        
        	
        
        
        
        	#loading saved weights
        	model.load_weights("model_vgg19_2nd_one.h5")


        	from keras.preprocessing.image import ImageDataGenerator
        
        	test_datagen = ImageDataGenerator(rescale=1./255)
        
        
        	itr1 = test_set1 = test_datagen.flow_from_directory(
                'static',
                target_size=(256, 256),
                batch_size=377,
                class_mode='categorical')
        
        	X1, y1 = itr1.next()
        	arr = model.predict(X1, batch_size=377, verbose=1)
        
        	arr = np.argmax(arr, axis=1)
			
			

           
        	"""
        	i = 0
        	j = 0
        	while(i < len(arr)):
        	  if(arr[i] == 1):
        	    j += 1
        	  i += 1
        	"""
	 
        #flash('Images with no alcohol content found: ' + j)
        
        #for layer in classifier.layers:
        #    g=layer.get_config()
        #    h=layer.get_weights()
        #    print (g)
        #    print (h)
        
        #scores = classifier.evaluate_generator(test_set,62/32)
        	flash(str(arr[0]))
        	if(arr[0] == 0):
				
        	  flash('Image has COVID disease ')
        	else:
			  	
        	  flash('Image does not have COVID disease ')
        
        	#K.clear_session()
        
        	K.clear_session()
        
        	os.remove('static/img/' + filename)
        	return render_template('Covid_frontend.html', user_image = 'static/img1/' + filename)
    return render_template('Covid_frontend.html')


if __name__ == '__main__':
    app.run(debug=True)
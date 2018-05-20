from PIL import Image
import numpy as np
import keras
from keras.models import model_from_json

basewidth = 28
baseheight = 28

file = "/root/7"

img = Image.open(file)
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), Image.ANTIALIAS)
hpercent = (baseheight / float(img.size[1]))
wsize = int((float(img.size[0]) * float(hpercent)))
img = img.resize((wsize, baseheight), Image.ANTIALIAS)
arr = np.asarray(img)
#print (arr.shape())

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=keras.optimizers.Adadelta(),
                     metrics=['accuracy'])
res = loaded_model.predict(arr)
print (res)
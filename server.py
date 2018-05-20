from flask import Flask, render_template, request
from keras.models import model_from_json
from keras import backend as K
from PIL import Image
import numpy as np
import keras
import os
# from future import print_function
import numpy
import json
from sklearn.model_selection import train_test_split
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
import glob
import json
import numpy
import os
import gzip
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/result', methods=['GET', 'POST'])
def result():
    select = request.form.get("letter");

    # Saving image into the picture directory

    img = request.files['img']
    file = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
    img.save(file)
    basewidth = 28
    baseheight = 28
    img = Image.open(file)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    hpercent = (baseheight / float(img.size[1]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((wsize, baseheight), Image.ANTIALIAS)
    img_array = np.array(img)
    img_array=img_array.reshape(1,28,28,1)

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model1 = model_from_json(loaded_model_json)
    loaded_model1.load_weights("model.h5")
    loaded_model1.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'])


    json_file = open('model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model2 = model_from_json(loaded_model_json)
    loaded_model2.load_weights("model2.h5")
    loaded_model2.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'])
#   res2 = loaded_model2.predict_classes(img_array)

    json_file = open('model3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model3 = model_from_json(loaded_model_json)
    loaded_model3.load_weights("model3.h5")
    loaded_model3.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'])
#    res3 = loaded_model.predict_classes(img_array)
    if (select == "1"):

        res1 = loaded_model1.predict_classes(img_array)
        res_str=np.array_str(res1)
        res_str=res_str.replace("[","")
        res_str = res_str.replace("]", "")
        K.clear_session()
        return render_template("index.html", final_text = "mnist",img=res_str)
    if (select == "2") :

        res2 = loaded_model2.predict_classes(img_array)
        res_str=np.array_str(res2)
        res_str=res_str.replace("[","")
        res_str = res_str.replace("]", "")
        K.clear_session()
        return render_template("index.html", final_text = "ua-mnist",img=res_str)


    if (select == "3") :
        res3 = loaded_model3.predict_classes(img_array)
        res_str=np.array_str(res3)
        res_str=res_str.replace("[","")
        res_str = res_str.replace("]", "")
        K.clear_session()
        return render_template("index.html", final_text = "fashio-mnist",img=res_str)



@app.route('/trainm', methods=['GET', 'POST'])
def train():

    dataset = request.form.get("pr")
    activation_function = request.form.get("parametr2")

    batch_size = int(request.form.get("value"))
    epochs = int(request.form.get("value2"))

    inputRange1 = int(request.form.get("input_range_1"))
    inputRange2 = int(request.form.get("input_range_2"))
    

    def SCRIPT_CONV_NEURAL_NETWORK(value, value2, pr, parametr2):


        data = json.load(open("config.json"))

        if (dataset == 1):
            if os.path.isfile(data["features_path"] + "/.npy"):  # Check if we already have numpy with our data
                dataset = numpy.load(data["features_path"] + "/.npy")
            else:
                dataset = read_data()  # Create and read numpy data (see data_reader.py)

            X = numpy.asarray([img[0] for img in dataset])
            y = numpy.asarray([img[2] for img in dataset])

            x_train, x_test, y_train, y_test = train_test_split(X, y)  # Divide our data on train and test samples

            img_rows, img_cols = 27, 35  # Set size of our images
            input_dim = 945
            num_classes = 72  # Write number of classes 72

        elif (dataset == 2):

            x_train, y_train = load_mnist(data["Mnist_dataset"], kind='train')
            x_test, y_test = load_mnist(data["Mnist_dataset"], kind='t10k')

            img_rows, img_cols = 28, 28  # Set size of our images
            input_dim = 784
            num_classes = 10

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation=activation_function,
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation=activation_function))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation=activation_function))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        return score

    def read_data():

        data = json.load(open("config.json"))  # Open config with path

        list_of_folders = glob.glob(data["dataset"][0] + "*")  # list of path of folders (ex. /home/.../letters/A_1)

        dataset = []

        for target in range(len(list_of_folders)):
            for path_to_img in glob.glob(list_of_folders[target] + "/*"):  #
                im = Image.open(path_to_img).convert('L')
                (width, height) = im.size
                greyscale_image = list(im.getdata())
                greyscale_image = numpy.array(greyscale_image)
                greyscale_image = greyscale_image.reshape((height, width))
                dataset.append((greyscale_image, path_to_img, target))
            target += 1

        numpy.save(data["features_path"], dataset)

        return dataset

    def load_mnist(path, kind='train'):


        """Load MNIST data from `path`"""
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

            return images, labels

    # if(selectValue1 == "2"):
    #     return render_template("training.html", how = "128")
    #return render_template("training.html", how=)
    read_data()
    load_mnist()
    score = SCRIPT_CONV_NEURAL_NETWORK(batch_size, epochs, dataset, activation_function)

    return render_template("training.html", how = str(score))











@app.route('/menux', methods=['GET', 'POST'])
def menux():
    submit = request.form.get("menux");
    if(submit == "1"):
        return render_template("index.html")
    if(submit == "2"):
        return render_template("training.html")
    if(submit == "3"):
        return render_template("how.html")
    if(submit == "4"):
        return render_template("aboutUs.html")




if __name__ == "__main__":
    app.run(debug=True)

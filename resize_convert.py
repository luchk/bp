from PIL import Image
import numpy as np
import glob


basewidth = 28
baseheight = 28

image_list = []
for infile in glob.glob("/Users/vitaliy_vorobyov/Desktop/*.jpg"):
    img = Image.open(infile)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    hpercent = (baseheight / float(img.size[1]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((wsize, baseheight), Image.ANTIALIAS)
    img.show()
    arr = np.array(img)
    image_list.append(arr)
# print(image_list)

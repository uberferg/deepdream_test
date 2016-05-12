import numpy as np
import matplotlib.pyplot as plt

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/christopher/Dropbox/2015-16PrincetonUniversity/COS495/Final Project/project_code/models/bvlc_googlenet/deploy.prototxt'
PRETRAINED = '/home/christopher/Dropbox/2015-16PrincetonUniversity/COS495/Final Project/project_code/models/bvlc_googlenet/bvlc_googlenet.caffemodel'


caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load('//home/christopher/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
#plt.imshow(input_image)

for imageName in ["baseball", "bedroom", "canyon", "city", "car", "chair", "dog", "forest_castle"]:
    input_image = caffe.io.load_image("/home/christopher/Dropbox/2015-16PrincetonUniversity/COS495/Final Project/project_code/input_files/priors/" + imageName + ".jpg")

    prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
    print 'prediction shape:', prediction[0].shape
    plt.plot(prediction[0])
    print 'predicted class:', prediction[0].argmax()
    plt.show()

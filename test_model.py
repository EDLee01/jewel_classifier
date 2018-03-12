from keras.models import load_model
import numpy as np
model = load_model('my_model.h5')

from keras.preprocessing import image
img = image.load_img('zuanshi.jpg',target_size=(100,100))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
p = model.predict_proba(x)
preds = model.predict_classes(x)
num = preds[0]
n = np.asscalar(np.int64(num))
print n

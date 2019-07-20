from flask import Flask
from flask import render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from flask import request

app = Flask(__name__)
def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(136, 102))
    img_tensor = image.img_to_array(img)          # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/evaluate", methods = ['POST'])
def evaluate():
     model = load_model("../model/save_last_complex_model1563649896.9996808.hdf5")
     request_file = request.files['file']
     img = load_image(request_file)
     pred = model.predict(img)
     print(pred)
     return "{code:true}"
if __name__ == "__main__":
    app.run(debug=True)


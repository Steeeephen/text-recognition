from flask import Flask, request, render_template, redirect

from core.utils import load_image
from core.image_annotator import ImageAnnotator

app = Flask(__name__)

image_annotator = ImageAnnotator()

@app.route('/')
def send_to_url():
  return redirect('/image')

@app.route('/image', methods=['GET', 'POST'])
def home():
  if request.files:

    image = request.files['image']
    file = image.filename
    image = load_image(image)

    image_annotator.annotate_image(image, file)
    return redirect(f'/image/{file}')
  else:
    return render_template('home.html')

@app.route('/image/<image_name>')
def show_image(image_name):
  return render_template('show_image.html', image_name=image_name)

app.run()
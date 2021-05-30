"""

this file holds the class used for annotating an image with the text present

Most of the code is a simplified version of both the CRAFT and deep-text-recognition-benchmark repos from Clova

"""
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms

from core.utils import *
from PIL import Image
from models.spotting.craft import CRAFT
from models.recognition.clova import CLOVA
from skimage import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageAnnotator():
    def __init__(self, 
                 spotter_model_path='models/spotting/craft.pth',
                 recogniser_model_path='models/recognition/recogniser.pth',
                 threshold=0.7,
                 spotting_image_size=1280,
                 spotting_image_ratio=1.5,
                 reading_image_height=32,
                 reading_image_width=100):
        """ 
        input:
            spotter_model_path: path to pretrained CRAFT model
            recogniser_model_path: path to pretrained TPS-Resnet-BiLSTM-Attn model
            threshold: confidence threshold for text spotting
            spotting_image_size: Canvas size to resize image for spotting
            spotting_image_ratio: Magnification ratio for resized image for spotting
            reading_image_height: Height of resized word image to read text from
            reading_image_width: Width of resized word image to read text from
        """
        self.threshold = threshold
        self.spotting_image_size = spotting_image_size
        self.spotting_image_ratio = spotting_image_ratio
        self.reading_image_width = reading_image_width
        self.reading_image_height = reading_image_height

        self.spotting_model = self.init_spotter(spotter_model_path)
        self.recogniser, self.converter = self.init_recogniser(recogniser_model_path)

    @staticmethod
    def init_spotter(path):
        spotting_model = CRAFT()

        spotting_model.load_state_dict(copy_state_dict(torch.load(path, map_location=device)))
        spotting_model.eval()

        return spotting_model

    @staticmethod
    def init_recogniser(path):
        recogniser = CLOVA()
        converter = recogniser.converter

        recogniser = torch.nn.DataParallel(recogniser).to(device)
        recogniser.load_state_dict(torch.load(path, map_location=device))

        recogniser.eval()

        return recogniser, converter

    def annotate_image(self, image, image_name):
        """
        Function to read the words in an image and annotate the image with the words.
        Image is saved to static/output/<image_name>

        input:
            image: Numpy array of image
            image_name: filename of the image
        output:
            None
        """
        # Resize Image
        img_resized, ratio = resize_aspect_ratio(
            image,
            self.spotting_image_size,
            self.spotting_image_ratio)

        ratio_h = ratio_w = 1 / ratio

        x = normalize_mean_variance(img_resized)

        # [height, width, channel] to [c, h, w] and then [batch, c, h, w]
        x = torch.from_numpy(x).permute(2, 0, 1).to(device)
        x = torch.autograd.Variable(x.unsqueeze(0))

        # forward pass
        with torch.no_grad():
            y, _ = self.spotting_model(x)

        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        boxes, polys = getDetBoxes(score_text, score_link, self.threshold, self.threshold, 0.4)

        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)

        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)

            poly_xmax, poly_ymax = poly.max(axis=0)
            poly_xmin, poly_ymin = poly.min(axis=0)

            border_ymax = np.min([image.shape[0], poly_ymax])
            border_xmax = np.min([image.shape[1], poly_xmax])

            border_ymin = np.max([0, poly_ymin])
            border_xmin = np.max([0, poly_xmin])

            borders = (
                slice(border_ymin, border_ymax),
                slice(border_xmin, border_xmax)
            )

            cropped = image[borders]

            word = self.read_text(cropped)
            image = np.asarray(image)

            cv2.putText(image, word, (border_xmin, border_ymin-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
            cv2.polylines(image, [poly], True, (255,0,0), 1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f'static/output/{image_name}', image)

        return

    def read_text(self, image):
        """
        Function to take the word image and pass it through the recognition model
        input:
          image: Numpy array denoting the image
        output:
          pred: String denoting predicted word in image
        """
        # Convert to PIL Image
        img = Image.fromarray(image).convert('L')

        # Transform image for input
        img = img.resize((self.reading_image_width, self.reading_image_height), Image.BICUBIC)
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        img = img.unsqueeze(0)

        # Load image to GPU if available, else CPU
        image = img.to(device)

        # For max length prediction
        length_for_pred = torch.IntTensor([25]).to(device)
        text_for_pred = torch.LongTensor(1, 26).fill_(0).to(device)

        # Pass through model
        preds = self.recogniser(image, text_for_pred)

        _, preds_index = preds.max(2)

        # Decode model output into text
        preds_str = self.converter.decode(preds_index, length_for_pred)

        # [s] denotes end of sentence token
        pred = preds_str[0].split('[s]')[0]

        return pred
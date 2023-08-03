import requests
from PIL import Image,ImageDraw
from io import BytesIO
import random
import os


def imread(path):
    if path.startswith('http') or path.startswith('https'):
        response = requests.get(path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(path).convert('RGB')
    return image

def random_image(root_path):
    img_list = os.listdir(root_path)
    img_item = random.sample(img_list, 1)[0]
    return Image.open(os.path.join(root_path, img_item))

def draw_points_to_image(image:Image.Image,points:list,radius=16,color = (255, 0, 0)):
    draw = ImageDraw.Draw(image)
    for [x,y] in points:
        draw.ellipse((x - radius, y - radius, x + radius,y + radius), fill=color)
    return image

def in_rectangle(bbox,points):
    for point in points:
        if min(max(point[0],bbox[0]),bbox[0]+bbox[2]) != point[0] or min(max(point[1],bbox[1]),bbox[1]+bbox[3]) != point[1] :
            return False
    
    return  True



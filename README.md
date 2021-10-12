# How to build a face tracker in Python

This project is highly inspired by this article from [Adrian Rosebrock](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/).
His website helps me a lot when it comes to learning Computer Vision.

## Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all requirements
```bash
pip install -r requirements.txt
```

## How does it work ?

The algorithm takes a video as input 
First, we detect faces using a HAAR cascade face detection algorithm.
This return boundinx boxes coordinates.
Then:
- Use bounding box coordinates and compute centroids
- Compute Euclidean distance between new bounding boxes and existing faces
- Update (x, y)-coordinates of existing faces
- Register new faces
- Deregister old faces

## Usage

For webcam usage
'''bash
python main.py
'''

For video
'''bash
python main.py -v your-video.mp4
'''

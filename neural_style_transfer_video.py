import time
import os
import subprocess
import cv2 


models_path = 'models/instance_norm/'
video_path = 'Tom_n_jerry.mp4'

def predict(img, h, w):
    blob = cv2.dnn.blobFromImage(img, 1.0, (w, h),
        (103.939, 116.779, 123.680), swapRB=False, crop=False)

    print ('Setting the input to the model')
    net.setInput(blob)

    print ('Starting Inference!')
    start = time.time()
    out = net.forward()
    end = time.time()
    print ('Inference Completed successfully!')

    # Reshape the output tensor and add back in the mean subtraction, and
    # then swap the channel ordering
    out = out.reshape((3, out.shape[2], out.shape[3]))
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.680
    out /= 255.0
    out = out.transpose(1, 2, 0)

    # Printing the inference time
    if False:
        print ('The model ran in {:.4f} seconds'.format(end-start))

    return out

def resize_img(img, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = img.shape[:2]

    if width is None and height is None:
        return img
    elif width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(img, dim, interpolation=inter)
    return resized


# Check if there are models to be loaded and list them
models = []
for f in sorted(os.listdir(models_path)):
    if f.endswith('.t7'):
        models.append(f)

if len(models) == 0:
    raise Exception('The model path doesn\'t contain models')


# Load the neural style transfer model
path = models_path + ('' if models_path.endswith('/') else '/')
print (path + models[2])
print ('Loading the model...')

model_loaded_i = -1
total_models = len(os.listdir(models_path))
model_loaded_i = 0
model_to_load = path + models[model_loaded_i]
net = cv2.dnn.readNetFromTorch(model_to_load)
print ('Model Loaded successfully!')

vid = cv2.VideoCapture(video_path)
while True:
    _, frame = vid.read()
    img = resize_img(frame, width=600)
    h, w  = img.shape[:2]
    out = predict(img, h, w)

    cv2.imshow('Stylizing Real-time Video', out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'): 
        model_loaded_i = (model_loaded_i + 1) % total_models
        model_to_load = path + models[model_loaded_i]
        net = cv2.dnn.readNetFromTorch(model_to_load)
    elif key == ord('p'):
        model_loaded_i = (model_loaded_i - 1) % total_models
        model_to_load = path + models[model_loaded_i]
        net = cv2.dnn.readNetFromTorch(model_to_load)

vid.release()
cv2.destroyAllWindows()

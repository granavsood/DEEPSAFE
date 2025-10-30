# utils.py
import torch
import numpy as np
import cv2
import face_recognition
from torchvision import transforms
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = torch.nn.Softmax(dim=1)

device = torch.device('cpu')  # <---- set device here
inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class ValidationDataset(Dataset):
    def __init__(self, video_path, sequence_length=20, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(self.video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def im_convert(tensor):
    """ Convert a tensor into a displayable image """
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = np.clip(image, 0, 1)
    return image

def predict(model, img):
    fmap, logits = model(img.to(device))
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100

    idx = np.argmax(logits.detach().cpu().numpy())
    bz, nc, h, w = fmap.shape
    out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T, weight_softmax[idx,:].T)
    predict_map = out.reshape(h, w)
    predict_map = predict_map - np.min(predict_map)
    predict_map = predict_map / np.max(predict_map)
    predict_map = np.uint8(255 * predict_map)

    heatmap = cv2.applyColorMap(cv2.resize(predict_map, (im_size, im_size)), cv2.COLORMAP_JET)
    img_orig = im_convert(img[:, -1, :, :, :])
    img_orig = (img_orig * 255).astype(np.uint8)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)

    # Blend heatmap and original image
    result = cv2.addWeighted(img_orig, 0.7, heatmap, 0.3, 0)

    # Convert BGR back to RGB for Streamlit
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return int(prediction.item()), confidence, result_rgb

import streamlit as st
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from mat_extract import descriptor_mat
from torch_geometric.data import Data
from extractor import ViTExtractor
from gnn_pool import GNNpool
import torch.optim as optim
from tqdm import tqdm
import util
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import njit
import urllib.request
import warnings
import math

def segment_image(uploaded_image):
    pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
    K = 2
    epoch = 10
    res = (224, 224)
    stride = 4
    facet = 'key'
    layer = 11
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_bin = False
    cc = False

    if not os.path.exists(pretrained_weights):
        url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
        util.download_url(url, pretrained_weights)
    uploaded_image = uploaded_image.convert('RGB')
    prep = transforms.Compose([
        transforms.Resize(res, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    image_tensor = prep(uploaded_image)[None, ...]
    image_np = np.array(uploaded_image)
    extractor = ViTExtractor('dino_vits8', stride, model_dir=pretrained_weights, device=device)
    feats_dim = 384
    model = GNNpool(feats_dim, 64, 32, K, device).to(device)
    torch.save(model.state_dict(), 'model.pt')
    model.train()
    W, F, D = descriptor_mat(image_tensor, extractor, layer, facet, bin=log_bin, device=device)
    node_feats, edge_index, edge_weight = util.load_data(W, F)
    data = Data(node_feats, edge_index, edge_weight).to(device)
    model.load_state_dict(torch.load('./model.pt', map_location=torch.device(device)))
    opt = optim.AdamW(model.parameters(), lr=0.001)
    for _ in range(epoch):
            opt.zero_grad()
            A, S = model(data, torch.from_numpy(W).to(device))
            loss = model.loss(A, S)
            loss.backward()
            opt.step()
    S = S.detach().cpu()
    S = torch.argmax(S, dim=-1)
    mask0, S = util.graph_to_mask(S, cc, stride, image_tensor, image_np)
    mask0_image = Image.fromarray(mask0 * 255).convert('L')
    # Convert the segmented mask to a numpy array
    mask0_np = np.array(mask0_image)

    # Convert the original image and mask to the same data type
    image_np = np.array(uploaded_image).astype(np.uint8)
    # Convert the segmented mask to a numpy array
    mask0_np = mask0_np.astype(np.uint8)

    # Create a color mask
    color_mask = np.zeros_like(image_np)
    segmented_areas = cv2.applyColorMap(mask0_np * 255, cv2.COLORMAP_JET)
    color_mask[mask0_np > 0] = segmented_areas[mask0_np > 0]

    # Overlay the color mask on the image
    overlay = cv2.addWeighted(image_np, 1.0, color_mask, 0.7, 0)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask0_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the segmented areas
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Convert the overlay image to a PIL Image
    overlay_image = Image.fromarray(overlay)

    return overlay_image
    # return mask0_image

# Streamlit UI
st.title('Crack Segmentation App 🚨')

# Introduction
st.write('**Welcome to the Crack Segmentation App!**')
st.write('**This app is designed to perform crack segmentation on images.**')

# Image upload
st.write('Upload an image for crack segmentation:')
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Add a button for the user to perform segmentation
    if st.button('Perform Segmentation'):
        # Perform crack segmentation
        image = Image.open(uploaded_image)
        segmented_image = segment_image(image)

        # Display the segmented image
        st.write('**Segmented Image:**')
        st.image(segmented_image, caption='Segmented Image', use_column_width=True)
st.text('')  # Add an empty line to separate sections
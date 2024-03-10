import streamlit as st
import pandas as pd
from PIL import Image
import glob as glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Define the path to the model file on your local machine
model_path = 'model14.pth'

def create_model(num_classes):
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device('cpu')


# load the model and the trained weights
model = create_model(num_classes=10).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

CLASSES = [
    'background', 'D00', 'D01', 'D10', 'D11', 'D20', 'D40', 'D43', 'D44', 'D50'
]

detection_threshold = 0.5
def predict(fid):
      local_image_path = './India/test/images/' + fid + '.jpg'

      # Read the local image
      image = cv2.imread(local_image_path)
      orig_image = image.copy()


      # BGR to RGB
      image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
      # Make the pixel range between 0 and 1
      image /= 255.0
      # Bring color channels to the front
      image = np.transpose(image, (2, 0, 1)).astype(float)
      # Convert to tensor
      image = torch.tensor(image, dtype=torch.float)
      # Add batch dimension
      image = torch.unsqueeze(image, 0)
      image = image.to(device)


      # Perform inference
      with torch.no_grad():
             outputs = model(image)

      # Load all detections to CPU for further operations
      outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
      # Carry further only if there are detected boxes
      if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # Filter out boxes according to detection_threshold
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # Get all the predicted class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            # Draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                  cv2.rectangle(orig_image,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 0, 255), 2)
                  cv2.putText(orig_image, pred_classes[j],
                              (int(box[0]), int(box[1]-5)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                              2, lineType=cv2.LINE_AA)

            # Display the image inline in VSCode
            plt.imshow(orig_image)
            # st.image(orig_image)
            plt.title('Prediction')
            plt.show()
            # Save the predicted image
            cv2.imwrite(f"./predicted_{local_image_path.split('/')[-1]}.jpg", orig_image)
            res_img="./predicted_"+local_image_path.split('/')[-1]+".jpg"
            st.image(res_img)




def road():
      st.title('Inspection of road ')
      img_file=st.file_uploader('Upload road image ',type=['png','jpg','jpeg'])
     

      if img_file is not None:
            fid=img_file.name.split('.')[0] 
            st.image(img_file,width=400)

      click=st.button("Inspect", type="primary")

      if not click:
            st.stop()
     
      # if click is not None:
      # uploaded_file = st.file_uploader("Upload XML file", type=["xml"])
      with st.empty():
            for percent in range(100):
                  time.sleep(.005)
                  st.progress(percent+1,text='processing')
            st.success('Processed')
      predict(fid)


      
     
     
           
     

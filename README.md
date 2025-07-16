# Osteoarthritis_CNN
Detecting and grading knee osteoarthritis from X-Ray images using Deep learning models.

Osteoarthritis Detection and Grading from Knee X-ray Images Using Deep Learning


Problem Statement

Osteoarthritis is a widespread degenerative joint disease that causes pain and reduced mobility, especially in elderly populations. Diagnosis typically relies on manual visual inspection of X-ray images, which can be time-consuming and subject to human error. Automating this process using machine learning can help in early detection, assist radiologists, and improve access to diagnostics in underserved regions. 

Objective/Goal

-To build a deep learning model that can classify the severity of osteoarthritis from knee X-ray images using MobileNetV2, and apply preprocessing to enhance image quality and region focus. 
- To build a deep learning model that detects the presence and severity (grade) of osteoarthritis from knee X-ray images.
- To create a simple web-based tool for uploading and analyzing knee X-rays. - Success Metrics: ≥85% accuracy for binary classification (OA vs. No OA); ≥70% accuracy for multi-class grading (KL Grade 0–4). 
-Perform Segmentation ( pointing out the region of interest). 



Dataset Used 

-We used the 'Knee Osteoarthritis Dataset with Severity' from Kaggle by Shashwat. It contains X-ray images of knees classified into 5 severity grades (0 to 4), distributed into train/test folders with respective labels. 

https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity
This dataset contains knee X-ray data for both knee joint detection and knee KL grading. The Grade descriptions are as follows:
●	Grade 0: Healthy knee image.
●	Grade 1 (Doubtful): Doubtful joint narrowing with possible osteophytic lipping
●	Grade 2 (Minimal): Definite presence of osteophytes and possible joint space narrowing
●	Grade 3 (Moderate): Multiple osteophytes, definite joint space narrowing, with mild sclerosis.
●	Grade 4 (Severe): Large osteophytes, significant joint narrowing, and severe sclerosis.

What we have done so far..
1. image_utils.py – Image Preprocessing
This script contains a function that prepares each X-ray image before it is used for training:
●	Loads the image in grayscale (removes color and focuses on structure).

●	Crops 60 pixels from the top and bottom to focus on the knee joint.

●	Resizes the image to (104, 224), then again to a square size of (224, 224) for compatibility with MobileNetV2.

●	Applies histogram equalization to enhance contrast.

●	Normalizes pixel values to the 0–1 range.

●	Converts grayscale image into a 3-channel (RGB-style) format by duplicating the single channel.

This preprocessing ensures that only relevant information (knee joint structure) is passed to the model, which improves classification accuracy.
________________________________________
2. data_loader.py – Dataset Loader
This script:
●	Iterates through the folders in the dataset directory (e.g., Grade_0, Grade_1...).

●	Each folder name ends with a number (e.g., “Grade_2” → label 2).

●	For every image inside the folder:

○	Calls preprocess_image() from image_utils.py.

○	Adds the processed image and its label to lists.

●	Returns all images and their corresponding labels as NumPy arrays.
This modular loading ensures you can easily preprocess and feed custom datasets into the model.
________________________________________
3. model_utils.py – Model Definition
Defines the MobileNetV2-based architecture used for classification:
●	Loads MobileNetV2 without the top classification layer (include_top=False) and with pretrained ImageNet weights.

●	Freezes the base initially to prevent large weight changes during early training.

●	Adds custom classification layers:

○	Global Average Pooling layer (reduces spatial dimensions).

○	Dense (fully connected) layer with 128 units and ReLU activation.

○	Dropout layer (30%) to reduce overfitting.

○	Final Dense layer with 5 output units and softmax activation to classify into grades 0–4.

This allows the model to reuse powerful low-level features from ImageNet and focus on learning OA-specific classification in later layers.
________________________________________
4. train_model.py – Training Pipeline
This is the main script where everything comes together.
Step-by-Step:
1.	 Loads and preprocesses all X-ray images using load_images_and_labels().

2.	 Splits data into training and validation sets (80/20 split).

3.	 Computes class weights to handle class imbalance.

4.	 Builds the MobileNetV2 model using build_mobilenet_model().

5.	 Sets up training callbacks:

○	EarlyStopping: stops training if validation loss doesn’t improve.

○	ReduceLROnPlateau: reduces learning rate if validation loss stagnates.

 Phase 1: Train Top Layers
●	Freezes MobileNetV2 base.

●	Trains only the new top layers (Dense + Dropout + Output).

●	Compiles and trains for 1 epoch using class weights.

 Phase 2: Fine-Tuning
●	Unfreezes the top layers of MobileNetV2 (from layer 100 onward).

●	Recompiles with a smaller learning rate (1e-5).

●	Trains again for 1 epoch to refine the full model.

 Final Steps:
●	Saves the model to disk.

●	Predicts on validation set and calculates performance metrics:

○	Classification report (precision, recall, F1-score).

○	Confusion matrix (displayed as heatmap).

●	Plots training and validation accuracy over epochs.

 By combining careful preprocessing (cropping + equalizing), transfer learning, and fine-tuning, this project improves the ability of the model to detect osteoarthritis severity accurately.


 
 

SEGMENTATION

Since our dataset do not have masks, we were not able to use U-net for segmentation.Segmentation is to show the area in the xray which might have caused the problem. For this , the dataset is supposed to have masks but since ours have mask , so  we are using an alternative method using Grad-Cam(Gradient-weighted Class Activation Mapping).
Using grad-cam we are going to give a segmentation like output which will give a accuracy close enough.
We are planning to integrate Grad-CAM with streamlit.
Here’s what your app is doing step-by-step:
1.	 Take a knee X-ray image (uploaded by the user)

2.	 Use your trained MobileNetV2 model to predict its KL grade

3.	 Apply Grad-CAM to compute the important region (heatmap)

4.	 Overlay that heatmap on the image to show which area influenced the prediction

5.	 Display it all inside a Streamlit web app.





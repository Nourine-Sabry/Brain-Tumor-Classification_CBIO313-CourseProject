# Brain Tumor Classification from MRI Images Using Deep Learning 🧠
## Project overview:
Brain tumors consist of cancerous or non-cancerous uncontrolled growth of abnormal cells in the central nervous system (CNS) that have no physiological function. They can be benign or malignant; both increase pressure in the brain as well as swelling. This eventually leads to a variety of unpleasant symptoms, ranging from personality changes to tremors and even death. Magnetic resonance imaging (MRI) is a brain imaging technique used to provide information on the type, size, position, and shape of a brain tumor, therefore it aids in diagnosis. With the help of deep learning, brain tumor classification techniques may help further classify the tumor into categories, such as the type of tumor present. The aim of this project is to implement a deep learning algorithm to classify MRI brain images into four categories: no tumor, glioma, meningioma, and pituitary. Therefore, answering the question “How accurately can a deep learning model identify and classify different types of tumors using MRI images?”.
## Link to dataset:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data?select=Training
## Dataset overview:
This dataset consists of testing and training data for the 4 groups: No tumor, pituitary, meningioma, and glioma. The testing data consists of around 300-400 images for each group separately. While the training data consists of 1300-1600 images for each group separately. Since neural networks require images to be of the same dimensions, this dataset requires cleaning (resizing), as the images are of different dimensions (length and width).
## Workflow:
> For a more detailed methodology, kindly check the PDF documentation file.
* Import libraries
* Load training and testing data
* Exploratory data analysis
* Split data into train, test, and validation
* Data resizing
* Build the model
* Train the model
* Visualize model performance
* Get predictions
* Model evaluation
## Made with:
* Tensorfow
* OS
* CV2
* Numpy
* Pandas
* Matplotlib
* Seaborn
## Author(s):
* Nourine Sabry

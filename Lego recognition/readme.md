# Lego Brick Classification Project
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/yourusername/yourrepo/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

![image](https://github.com/user-attachments/assets/e583f8b0-8545-4382-a374-bcca8b35f4f7)

## Table of Contents
- [Introduction](#introduction)
- [Team Composition](#team-composition)
- [Project Description](#project-description)
  - [Motivation](#motivation)
  - [Project Assumptions](#project-assumptions)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
  - [Requirements](#requirements)
  - [Running on Google Colab](#running-on-google-colab)
  - [Running Locally on CPU](#running-locally-on-cpu)
  - [Running Locally on GPU](#running-locally-on-gpu)
- [Methodology](#methodology)
  - [Data Processing](#data-processing)
  - [Model Design](#model-design)
    - [RocketNet](#rocketnet)
    - [AlexNet](#alexnet)
- [Training and Evaluation](#training-and-evaluation)
  - [Training Details](#training-details)
  - [Evaluation Results](#evaluation-results)
- [Grad-CAM Visualization](#grad-cam-visualization)
- [Applications](#applications)
- [Conclusion and Acknowledgments](#conclusion-and-acknowledgments)
- [References](#references)

---

## Introduction

This project focuses on building a convolutional neural network (CNN) model capable of recognizing individual LEGO bricks from images. Leveraging deep learning techniques, specifically CNNs, the model classifies LEGO bricks into different categories. The project was undertaken as part of a Data Science bootcamp and aims to contribute to applications that require LEGO brick recognition.

## Team Composition

- **Aleksandra Baran** - [LinkedIn](http://linkedin.com/in/alexabaran) | [GitHub](https://github.com/alexabaran)
- **Dariusz Balcer** - [LinkedIn](https://www.linkedin.com/in/dariuszbalcer/) | [GitHub](https://github.com/montenegro-db)
- **Grzegorz Gomza** - [LinkedIn](https://www.linkedin.com/in/gregory-gomza/) | [GitHub](https://github.com/grzegorz-gomza/)

**Team Mentor/Supervisor:**

- **Mateusz Maj** - [LinkedIn](https://www.linkedin.com/in/mateusz-maj-data-scientist/)

## Project Description

### Motivation

LEGO is one of the world's most recognizable toys, beloved by children and adults alike. The motivation for this project came from a scenario presented by a fictional stakeholder—a startup creator working on an app to recognize LEGO bricks based on user photographs. Such a system could be invaluable to LEGO collectors and enthusiasts, enabling quick identification of individual pieces.

### Project Assumptions

The primary goal was to create a model that allows the recognition of individual LEGO elements from photos. Specifically, we aimed to:

- Use a convolutional neural network (CNN) for image classification.
- Train the model on a dataset of LEGO brick images.
- Create a model capable of recognizing 20 different types of LEGO bricks.
- Overcome hardware and resource limitations by optimizing data processing and model design.

## Dataset

We used a dataset available on the [Kaggle platform](https://www.kaggle.com/datasets/ronanpickell/b200c-lego-classification-dataset), which includes a wide collection of LEGO images of various shapes and sizes.

- **Dataset Name**: B200C LEGO Classification Dataset
- **Number of Classes**: 20
- **Images per Class**: Approximately 4,000

Due to hardware limitations, we focused on a subset of the catalog, consisting of 20 different bricks, with 4,000 images for each brick. This choice allowed us to manage computational resources effectively while still providing valuable insights.

## Environment Setup

There are three ways to run the code in this project:

1. **Google Colab**
2. **Locally on your machine using CPU**
3. **Locally on your machine using GPU**

### Requirements

- **Python**: 3.10 or higher
- **Libraries**:
  - TensorFlow
  - Keras
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - Pickle-mixin
  - Kaggle
  - Jupyter
  - Pillow
  - Python-dotenv

### Running on Google Colab

Google Colab is a free cloud-based service that provides CPU and GPU resources.

1. **Open the Notebook**

   - Upload the `Lego_DL.ipynb` script to your Google Drive or create a new notebook and copy the code.

2. **Install Required Libraries**

   At the beginning of your Colab notebook, run the following cell to install required libraries (most of them are preinstalled on Colab):

   ```python
   !pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn pickle-mixin kaggle Pillow python-dotenv
   ```

3. **Set Up Kaggle API**

   - Upload your `kaggle.json` file to Colab:

     ```python
     from google.colab import files
     files.upload()
     ```
      or mount the google drive with code:
    
      ```python
        from google.colab import drive
        drive.mount('/content/drive')
      ```
      and move the files manualy via explorer in Google Colab

   - Move `kaggle.json` to the appropriate directory named ".kaggle"

4. **Download the Dataset**

   - Use the Kaggle API to download the dataset:

     ```python
     !kaggle datasets download -d ronanpickell/b200c-lego-classification-dataset -p ./data/lego-dataset --unzip
     ```
      or execute the first code blocks in the Lego_DL.ipynb notebook

   
6. **Run the Code**

   - Execute the cells in the notebook to run the code.

### Running Locally on CPU

To run the code locally on your machine using CPU:

1. **Install Python and Pip**

   Ensure you have Python 3.10 or higher and pip installed on your system.

2. **Set Up Virtual Environment**

   Create and activate a virtual environment:

   ```bash
   python3 -m venv env
   # On Unix or MacOS
   source env/bin/activate
   # On Windows
   env\Scripts\activate
   ```

3. **Install Required Libraries**

   Install the necessary Python libraries using the following `pip` command:

   ```bash
   pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn pickle-mixin kaggle jupyter Pillow python-dotenv
   ```

   > **Note**: For more details on installing TensorFlow, refer to the [TensorFlow installation guide](https://www.tensorflow.org/install/pip).
   > Due to updates in libraries, it is possible that the names of libraries would be changed. For TensorFlow, it might be tensorflow-cpu instead of tensorflow.

4. **Set Up Kaggle API**

    Please check the documentation at [Kaggle API](https://www.kaggle.com/docs/api)
   
   - Place your `kaggle.json` file in the directory `~/.kaggle` (on Unix systems) or `%USERPROFILE%\.kaggle` (on Windows).

     ```bash
     mkdir ~/.kaggle
     mv /path/to/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

   - Ensure that the `KAGGLE_CONFIG_DIR` environment variable is set appropriately, or set it in a `.env` file in your project's root directory.

5. **Download the Dataset**

   Run the first cells in `Lego_DL.ipynb` script to download the dataset

### Running Locally on GPU

To run the code locally using GPU acceleration, you need to set up your environment with the necessary GPU drivers and libraries.

1. **Install CUDA and cuDNN**

   - Install the NVIDIA CUDA Toolkit and cuDNN library.

     - **CUDA Toolkit**: [CUDA Toolkit Documentation](https://developer.nvidia.com/cuda-toolkit)
     - **cuDNN**: [cuDNN Documentation](https://developer.nvidia.com/cudnn)

   Ensure that your GPU is compatible and that the appropriate drivers are installed.

   > **Note**: Installing CUDA and cuDNN can be complex. For detailed instructions, refer to the official documentation.

2. **Install Rapids.ai**

   We recommend installing Rapids.ai, which provides a suite of GPU-accelerated libraries and includes TensorFlow with necessary packages, including CUDA.

   - **Rapids.ai Installation Guide**: [Install Rapids.ai](https://rapids.ai/start.html)

   - Install Rapids.ai using conda (recommended) - plase use the code generator provided in the documentation

3. **Install Required Libraries**

   Install the additional necessary Python libraries:

   ```bash
   pip numpy pandas matplotlib seaborn scikit-learn pickle-mixin kaggle jupyter Pillow python-dotenv
   ```

4. **Set Up Kaggle API**

   - Follow the same steps as before to set up the Kaggle API.

5. **Download the Dataset and Run the Code**

    Run the first cells in `Lego_DL.ipynb` script to download the dataset

   > **Note**: For detailed instructions on setting up TensorFlow with GPU support, refer to the [TensorFlow GPU installation guide](https://www.tensorflow.org/install/gpu).

6. **Additional Considerations**

   - If you're on Windows, you may need to use WSL (Windows Subsystem for Linux) to run the code with GPU support.

     - **WSL Documentation**: [Install WSL](https://docs.microsoft.com/en-us/windows/wsl/install)

   - Ensure that the NVIDIA drivers are properly installed and accessible within WSL.

   - For detailed setup instructions, refer to the [WSL and NVIDIA CUDA documentation](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

---

## Methodology

### Data Processing

We manually processed the images to create training, validation, and test sets. We did not use Keras's `ImageDataGenerator` or `dataset` classes, which resulted in faster code execution.

1. **Listing Image Paths**

   We generated a list of paths to all image files in the dataset:

   ```python
   def list_paths(folder):
       # Generates a list of paths to all files in a given folder
   ```

2. **Sampling and Class Selection**

   We took a random sample of images based on a given percentage or selected specific classes:

   ```python
   def process_data(folder, percent, take_sample_of, num_classes=None, folder_deep=0):
       # Processes data by taking a sample or selecting classes
   ```

3. **Creating Data and Labels**

   We manually loaded images and extracted labels from the directory structure:

   ```python
   def create_data_and_labels(image_paths, val_size=0.2, test_size=0.2):
       # Creates a dataset of images and their corresponding labels
   ```

4. **Data Splitting**

   We split the data into training, validation, and test sets using `train_test_split` from scikit-learn.

5. **One-Hot Encoding**

   We performed one-hot encoding on labels using `LabelEncoder` and `OneHotEncoder`:

   ```python
   label_encoder = LabelEncoder()
   y_train_encoded = label_encoder.fit_transform(y_train)
   # ...
   onehot_encoder = OneHotEncoder(sparse_output=False)
   y_train_cat = onehot_encoder.fit_transform(y_train_encoded.reshape(-1, 1))
   ```

### Model Design

#### RocketNet

RocketNet is a custom CNN architecture designed specifically for this project.

- **Architecture**:

  - **Input Layer**: Accepts 64x64x3 images.
  - **Convolutional Layers**:
    - Three convolutional layers with 32, 64, and 128 filters.
    - ReLU activation function.
    - Batch normalization and max-pooling layers between convolutional layers.
  - **Flatten Layer**: Transitions from convolutional layers to dense layers.
  - **Dense Layers**:
    - One dense layer with 512 neurons.
    - Dropout layer with a 50% rate to prevent overfitting.
  - **Output Layer**:
    - 20 neurons corresponding to the number of classes.
    - Softmax activation function for multi-class classification.

#### AlexNet

We implemented a scaled-down version of the [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) architecture suitable for 64x64x3 images.

- **Architecture**:

  - **Convolutional Layers**:
    - Five convolutional layers with varying filter sizes and strides.
    - ReLU activation function.
    - Batch normalization and max-pooling layers.
  - **Dense Layers**:
    - Two dense layers with 1024 and 512 neurons.
    - Dropout layers with a 50% rate after each dense layer.
  - **Output Layer**:
    - 20 neurons corresponding to the number of classes.
    - Softmax activation function.

## Training and Evaluation

### Training Details

- **Optimizer**: Adam with a learning rate of 0.001.
- **Loss Function**: Categorical Crossentropy.
- **Metrics**: Categorical Accuracy and AUC.
- **Early Stopping**: Monitored validation AUC with a patience of 5 epochs.

We trained both models on the training set and validated on the validation set. Early stopping was used to prevent overfitting.

Below are details regarding training of **AlexNet**:  
![image](https://github.com/user-attachments/assets/4aac0f2d-61a4-4d41-b252-f2ff92675897)  
  
and here is **RocketNet**:
![image](https://github.com/user-attachments/assets/3790ae4d-61f7-4283-b1ee-6cc4fbdae67b)



### Evaluation Results

- **RocketNet**:

  - Achieved an accuracy of approximately **88.9%** on the validation set.
  - Showed better generalization by distributing errors among all classes rather than concentrating on a few.

- **AlexNet**:

  - Achieved an accuracy of approximately **90.9%** on the validation set.
  - Had concentrated errors on specific classes.

Based on these results, we conclude that simpler networks like RocketNet can perform comparably to more complex architectures like AlexNet while offering better generalization.

- **Disclaimer**:
  The Legobricks are taken randomly by creating the dataset with the functions provided in the code. The results may differ because other brick types were used for the training in comparison to the provided conclusions and results. The purpose is mainly to learn by doing, so the 100% reproducibility is not the main focus in this project.

## Grad-CAM Visualization

To gain a deeper understanding of what goes on in the process of image recognition through a convolutional network, we used the **Grad-CAM** technique to visualize the activation maps.

Grad-CAM helps in:

- **Visualizing Important Regions**: Highlighting areas of the image that the model focuses on when making predictions.
- **Understanding Model Decisions**: Providing insights into how the model interprets different visual features.

We visualized the heatmaps generated at different convolutional layers and overlaid them on the original images. This analysis revealed how each model processes images and which features are important for classification.

Below are examples of Grad-CAM visualization generated by our models:
![Model_AlexNet_image_27](https://github.com/user-attachments/assets/b9bf89e8-3b90-4c99-beef-6a85eff79016)


## Applications

The models developed have potential applications in various scenarios:

1. **LEGO Building App**:

   - **Stack of Bricks Analysis**: Scans available bricks and recognizes all elements.
   - **Model Suggestions**: Recommends models that can be built with the available bricks.
   - **Automatic Generation of Instructions**: Provides step-by-step building instructions.

2. **App to Support Blind and Visually Impaired People**:

   - **Project Selection**: Offers accessible project descriptions.
   - **LEGO Brick Recognition**: Uses camera-based recognition with voice prompts.
   - **Learning Braille**: Integrates with Braille learning kits for educational purposes.
   - **Voice Instructions**: Guides users through building steps.

3. **Collection Management App**:

   - **Automatic Block Recognition**: Identifies shape, color, and features of bricks.
   - **Categorization**: Sorts blocks based on various attributes.
   - **Support for Companies**: Assists in sorting large quantities of mixed bricks.

4. **LEGO Shopping and Exchange App**:

   - **Scanning and Identifying Blocks**: Catalogs items for efficient management.
   - **Finding Missing Items**: Identifies missing pieces in sets.
   - **Shopping Hints**: Compares prices and offers the best purchase options.
   - **Brick Swapping Platform**: Enables exchange between collectors.

## Conclusion and Acknowledgments

We are extremely satisfied with the results of our project and the course of the entire project. Despite hardware limitations, we successfully developed functional models with potential for further development. Our work demonstrates that simpler architectures can perform as effectively as more complex ones, sometimes offering better generalization.

**Acknowledgments**:

We would like to express our immense appreciation to **Mateusz Maj**, our course supervisor, who oversaw the entire project. His support was invaluable not only in the context of this project but also in previous projects carried out during the course. His guidance and willingness to help were crucial to our success throughout the bootcamp.

**Team Reflection**:

The work on the project was not only an opportunity for our team to develop technically but also a chance for great collaboration. Each member brought unique experiences and skills, allowing us to explore issues from different perspectives. We are extremely satisfied with the results and hope to continue collaborating on future projects.

---

## References

- [Kaggle B200C LEGO Classification Dataset](https://www.kaggle.com/datasets/ronanpickell/b200c-lego-classification-dataset)
- [TensorFlow Installation Guide](https://www.tensorflow.org/install/pip)
- [TensorFlow GPU Installation Guide](https://www.tensorflow.org/install/gpu)
- [NVIDIA CUDA Toolkit Documentation](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN Documentation](https://developer.nvidia.com/cudnn)
- [Rapids.ai](https://rapids.ai/start.html)
- [WSL Documentation](https://docs.microsoft.com/en-us/windows/wsl/)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

---

Feel free to explore the code in [`Lego_DL.ipynb`](Lego_DL.ipynb) for detailed implementation. Contributions and feedback are welcome!

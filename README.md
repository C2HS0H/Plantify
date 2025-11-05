# Plantify 🌿

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)![Framework](https://img.shields.io/badge/Flask-2.0-black?style=for-the-badge&logo=flask&logoColor=white)![ML Library](https://img.shields.io/badge/PyTorch-1.9-orange?style=for-the-badge&logo=pytorch&logoColor=white)![Frontend](https://img.shields.io/badge/Tailwind_CSS-3.0-blueviolet?style=for-the-badge&logo=tailwind-css&logoColor=white)![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

**AI-Powered Plant Disease Detection using Deep Learning**

Plantify is an advanced AI-powered application designed to assist farmers and gardeners in identifying plant diseases affecting their crops. By leveraging deep learning techniques, the platform analyzes uploaded images of plant leaves using a Convolutional Neural Network (CNN) classifier. The backend server processes these images, accurately classifying both the crop and any detected diseases. Once identified, the application provides detailed insights, including disease information, prevention strategies, and recommended solutions, empowering users to take timely and effective action.

## ⭐ Features

- **Comprehensive Disease Coverage:** Identifies **39 different categories** of plant diseases and healthy leaves.
- **Wide Plant Variety:** Supports diagnosis for **16 distinct plant species**, including Apple, Corn, Grape, and Tomato.
- **High-Accuracy CNN Model:** Utilizes a powerful Convolutional Neural Network built with PyTorch for precise and reliable classification.
- **Instantaneous Predictions:** Delivers real-time analysis of uploaded leaf images, providing immediate feedback.
- **Actionable Solutions:** Offers detailed descriptions, prevention steps, and direct links to purchase recommended supplements or fertilizers.
- **Modern & Responsive UI:** Built with Tailwind CSS for a clean, intuitive, and mobile-friendly user experience.

---

## 🛠️ Technologies Used

- **Backend:** Python, Flask
- **Machine Learning:** PyTorch, NumPy, Pandas, Pillow
- **Frontend:** HTML5, Tailwind CSS, JavaScript
- **Deployment:** Gunicorn, Heroku (or any WSGI server)

---

## 🚀 Getting Started

Follow these steps to run the project on your local machine.

### Prerequisites

- Python 3.8 or higher
- A virtual environment tool (like `venv`)

### Installation

1.  **Clone the project repository:**

    ```sh
    git clone https://github.com/C2HS05/Plantify.git
    ```

2.  **Navigate to the project directory:**

    ```sh
    cd Plantify
    ```

3.  **Create and activate a virtual environment:**

    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

4.  **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

5.  **Download the pre-trained model:**

    - Download the model file `model.pt` from [here](https://gofile.io/d/ZJmoF2).

6.  **Run the Flask application:**
    ```sh
    python3 app.py
    ```
    Your application should now be running on `http://127.0.0.1:5000/`.

---

## 🧠 The Model

The core of Plantify is a custom-built Convolutional Neural Network (CNN) trained to classify plant leaf images.

### 📊 Dataset & Augmentation

The model was trained on the **PlantVillage Dataset**, a comprehensive, open-source collection of plant leaf images.

- The dataset contains **61,486 images** across **39 classes** of plant leaves and backgrounds.
- To enhance the model's robustness and prevent overfitting, we employed six different **data augmentation** techniques to artificially increase the dataset size:
  - Image Flipping
  - Gamma Correction
  - Noise Injection
  - PCA Color Augmentation
  - Rotation
  - Scaling

### ⚙️ Model Architecture Explained

For those new to CNNs, here’s a simplified breakdown of how the model processes an image:

1.  **Input Processing:** All uploaded images are resized to a standard `224x224` pixels with 3 color channels (RGB) before being fed into the network.

2.  **Feature Extraction (Convolution):** The image is passed through a series of convolutional layers. The first layer applies **32 filters** (kernels) to the image, each designed to detect basic features like edges, curves, and textures. This process generates a **feature map**, transforming the input from `3x224x224` into `32x224x224`.

3.  **Activation & Normalization:** A **ReLU** activation function is applied to introduce non-linearity, allowing the model to learn more complex patterns. **Batch Normalization** then normalizes the data to stabilize and accelerate the training process.

4.  **Down-sampling (Max Pooling):** A **Max Pooling** layer reduces the size of the feature map (e.g., to `32x112x112`), retaining only the most prominent features detected. This makes the model more efficient and helps it focus on the most important information.

5.  **Repetition:** This sequence of convolution, activation, normalization, and pooling is **repeated multiple times**. With each repetition, the layers learn progressively more complex and abstract features (from simple edges to parts of a leaf and eventually disease patterns).

6.  **Classification (Fully Connected Layers):** The final feature map is **flattened** into a long one-dimensional vector. This vector is then fed into a series of **fully connected (or dense) layers**, which analyze the combination of all detected features to make a final decision.

7.  **Final Prediction:** The final layer outputs a tensor of **39 scores**, one for each possible class. The model's prediction is the class corresponding to the **highest score** in the tensor. This index is then mapped to a human-readable disease name to be displayed to the user.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` file for more information.

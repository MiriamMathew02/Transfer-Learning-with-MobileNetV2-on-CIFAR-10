# Transfer Learning with MobileNetV2 on CIFAR-10
This project implements transfer learning using the MobileNetV2 architecture in TensorFlow, achieving 85% accuracy on the CIFAR-10 dataset.
## Features
- **Transfer Learning:** Utilizes MobileNetV2 pretrained on ImageNet for feature extraction and fine-tuning on CIFAR-10.
- **Model Optimization:**
  - Applied **data augmentation** to enhance generalization.
  - Used a **learning rate scheduler** to improve convergence.
- **Deployment:** The model is deployed using Flask with a minimalistic front-end built using HTML and CSS.
- **Real-Time Predictions:** Allows users to upload images via the web interface and predicts the class in real-time.

---

## Dataset
The project uses the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which consists of 60,000 32x32 color images across 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mobilenetv2-cifar10-flask.git
   cd mobilenetv2-cifar10-flask
   ```
2. Set up the environment:
 - Install the required dependencies
3. Download the CIFAR-10 dataset
4. Run the application:
   ```
   python app.py
   ```
## Deployment
- The app.py script uses Flask to serve the trained model.
- The front-end interface is built with HTML and CSS for simplicity and usability.
  
## Key techniques applied:
- Data Augmentation: Random flips, rotations, and zoom.
- Learning Rate Scheduler: Reduces the learning rate dynamically to ensure smooth convergence.

# Brain Tumor Detection Using Deep Learning 🧠💻

A deep learning-based application to detect brain tumors from MRI scans using a Convolutional Neural Network (CNN) built on the VGG19 architecture. The model predicts whether a tumor is present in the uploaded MRI image and provides the accuracy of each individual prediction.

<h2>🌟 Project Overview</h2>
Brain tumor detection is a crucial aspect of medical imaging, aimed at assisting doctors and radiologists in diagnosing conditions accurately. This project leverages deep learning, specifically Convolutional Neural Networks (CNN), to analyze MRI images and identify the presence of tumors.

<h3>Key Features</h3>
<ul>
 <li>Real-time Image Upload & Prediction:<br>
  Users can upload MRI images, and the model provides immediate feedback.
 </li>
 <li>Accuracy Reporting:<br>
  Displays the prediction probability for each image, enhancing trust and transparency.
 </li>
 <li>
  VGG19 Architecture: <br>
  Fine-tuned for optimized performance in medical image classification.
 </li>
</ul>

<h3>📚 Technologies Used</h3>
<ul>  
<li> Python: <br> For building the backend and processing data.</li>
<li>  Flask:<br> Lightweight web framework for serving the application.</li>
<li> TensorFlow and Keras:<br> For constructing and training the deep learning model.</li>
<li> OpenCV & PIL:<br> To handle and preprocess MRI images.</li>
 <li> HTML/CSS/JavaScript: For the frontend interface, providing a user-friendly experience.</li>
</ul>

<h3>⚙️ File Structure</h3>
<img src="https://github.com/Sushant2253/Brain-Tumor-Detection/blob/main/folder_structure.png" alt="Folder Structure" width="850" height="600">


<h3>🧠 About Convolutional Neural Networks (CNN)</h3>
<p>Convolutional Neural Networks (CNNs) are deep learning architectures commonly used for image processing tasks. CNNs are particularly effective in image classification due to their ability to capture spatial hierarchies in images.</p>

<p>In this project, we use the VGG19 architecture, a pre-trained model known for its depth and performance in image recognition tasks. VGG19 employs 19 layers with learnable weights and is effective in capturing fine-grained details, making it suitable for MRI image analysis.</p>

For a deeper understanding of CNNs, refer to this guide on CNNs.<br>
For more on VGG19, explore VGG19 Architecture.<br>

<h3>🚀 How to Use the Application</h3>
<h4>Prerequisites</h4>
Install Python (>=3.6)<br>
Install required dependencies:<br>
pip install -r requirements.txt<br>
Run the Application<br>
Start the Flask Server:<br>
python app.py<br>
Access the Application: Open a browser and go to http://127.0.0.1:5000/<br>
Usage<br>
Upload MRI Image: Select an MRI image file (formats: .png, .jpg, .jpeg).<br>
Predict: Click on the “Predict” button.<br>
View Results: The app will display the result (either "Yes Brain Tumor" or "No Brain Tumor") <br>
 
<h3>📊 Model Details</h3>
Base Model: VGG19 without the top classification layer, fine-tuned for this project.<br>
Layers: Added custom layers for improved classification accuracy:<br>
Flatten Layer: To reshape the output from convolution layers.<br>
Dense Layers: 4608 and 1152 neurons with ReLU activation, followed by a final dense layer with softmax activation for binary classification.<br>
Accuracy: The accuracy of each prediction is computed and displayed after image processing.<br>

<h3>🎨 Frontend Design</h3>
The user interface is designed to be minimalistic yet informative, providing users with easy access to upload and prediction functions. Key features include:<br>

Image Preview: Users can view the uploaded image before prediction.<br>
Dynamic Feedback: Prediction result and accuracy are displayed post-upload, enhancing transparency.<br>
<hr>
<h3>📞 Contact</h3><br>
For any inquiries, please contact:<br>

Email: sushantumap1234@gmail.com<br>

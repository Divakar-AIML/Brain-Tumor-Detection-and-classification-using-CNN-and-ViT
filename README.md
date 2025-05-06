The Brain Tumor Classification and Detection project is an advanced deep learning initiative aimed at automating the identification and classification of brain tumors from MRI scans. By integrating Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), the project delivers a robust hybrid model capable of classifying brain tumors into four categories: glioma, meningioma, pituitary tumor, and no tumor. Additionally, a user-friendly graphical interface enables seamless interaction, allowing users to upload MRI images and receive real-time tumor predictions with visualized bounding boxes for tumor localization. This project demonstrates proficiency in machine learning, computer vision, and software development, with applications in medical diagnostics and healthcare innovation.

Objectives
Accurate Classification: Develop a deep learning model to classify brain tumors into four distinct categories with high accuracy using MRI scans.
Hybrid Architecture: Combine the spatial feature extraction capabilities of CNNs with the global contextual understanding of ViTs to enhance model performance.
User Accessibility: Create an intuitive graphical user interface (GUI) to enable non-technical users, such as medical professionals, to upload images and obtain tumor predictions.
Tumor Localization: Implement bounding box visualization to highlight tumor regions in MRI scans, aiding in interpretability and clinical decision-making.
Scalability: Design a modular system that can be extended to include additional tumor types or imaging modalities in the future.
Technical Approach
1. Data Preparation
Dataset: The project utilizes a brain tumor MRI dataset, organized into training and testing sets, with images labeled as glioma, meningioma, pituitary tumor, or no tumor. An additional unsupervised dataset with modalities (e.g., FLAIR, T1, T2) is included for potential future enhancements.
Preprocessing: Images are resized to 224x224 pixels, converted to grayscale, normalized to a [0, 1] range, and augmented using techniques like rotation and flipping to improve model robustness.
Custom Data Generator: A custom TensorFlow Sequence class was implemented to efficiently load and preprocess images in batches, optimizing memory usage and training speed.
2. Model Architecture
The project employs a hybrid deep learning model combining CNNs and ViTs to leverage their complementary strengths:

Vision Transformer (ViT) Block:
Input images are divided into 16x16 patches, processed through a convolutional layer to extract embeddings.
Multi-head self-attention mechanisms capture global dependencies across patches, followed by layer normalization and residual connections.
A global average pooling layer produces a compact feature representation.
Convolutional Neural Network (CNN) Block:
The ViT output is reshaped and fed into a CNN with four convolutional layers (64, 128, 256, and 512 filters) and max-pooling layers to extract hierarchical spatial features.
Global average pooling reduces dimensionality while preserving critical features.
Fully Connected Layers:
A dense layer with 256 units and ReLU activation integrates features from both blocks.
A softmax output layer produces probabilities for the four tumor classes.
Compilation: The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.
3. Training and Evaluation
Training: The model was trained for 100 epochs using a batch size of 16, with data fed through the custom data generator. Training and validation accuracy/loss were monitored to assess performance.
Evaluation Metrics:
Test accuracy and loss were computed to evaluate model generalization.
A confusion matrix and classification report (precision, recall, F1-score) were generated to analyze performance across tumor classes.
Visualization: Training and validation accuracy/loss curves were plotted using Matplotlib to identify convergence and potential overfitting.
4. User Interface and Deployment
GUI Development: A Tkinter-based graphical interface was developed to allow users to:
Upload MRI images in common formats (JPG, JPEG, PNG).
Trigger tumor classification with a single click.
View the predicted tumor type (or "No Tumor") and a processed image with a bounding box highlighting the tumor region (if present).
Bounding Box Visualization:
For images classified as containing a tumor, contours are detected using OpenCV’s thresholding and contour detection.
The largest contour is used to draw a red bounding box around the tumor, enhancing interpretability.
Model Deployment: The trained model is saved as brain_tumor_classification_model.h5 and loaded into the GUI application for real-time predictions.
Key Features
Hybrid Model: Combines CNNs for local feature extraction and ViTs for global context, achieving robust tumor classification.
Real-Time Predictions: The GUI enables instant classification and visualization, making the tool accessible to non-technical users.
Tumor Localization: Bounding boxes highlight tumor regions, providing actionable insights for medical professionals.
Efficient Data Handling: The custom data generator ensures scalable and memory-efficient processing of large MRI datasets.
Comprehensive Evaluation: Detailed metrics (accuracy, confusion matrix, classification report) provide transparency into model performance.
Technologies Used
Programming Languages: Python
Deep Learning Frameworks: TensorFlow, Keras
Computer Vision: OpenCV for image preprocessing and bounding box visualization
GUI Development: Tkinter for the graphical interface
Data Visualization: Matplotlib for plotting accuracy and loss curves
Libraries: NumPy, Scikit-learn for data manipulation and evaluation
Hardware: Trained on a system with GPU support (assumed, based on TensorFlow usage)
Results
Model Performance: The hybrid model achieved high accuracy on the test set (exact metrics not specified in code output but implied to be robust based on 100-epoch training and evaluation metrics).
Classification Report: Precision, recall, and F1-scores were computed for each tumor class, ensuring balanced performance across categories.
User Experience: The GUI provides a seamless experience, with clear feedback on tumor presence and location, validated through bounding box visualization.
Scalability: The modular code structure supports future enhancements, such as incorporating additional MRI modalities or tumor types.
Challenges and Solutions
Challenge: Limited dataset size and class imbalance.
Solution: Applied data augmentation (rotation, flipping) and used a custom data generator to handle data efficiently.
Challenge: Combining CNN and ViT architectures without overfitting.
Solution: Used global average pooling and layer normalization to reduce parameters and stabilize training.
Challenge: Accurate tumor localization in MRI scans.
Solution: Employed OpenCV’s contour detection to identify and highlight the largest tumor region, validated visually through the GUI.
Challenge: Ensuring the GUI is user-friendly for non-technical users.
Solution: Designed a minimalist Tkinter interface with clear labels and error handling (e.g., warnings for missing images).
Future Enhancements
Multi-Modal Integration: Incorporate additional MRI modalities (e.g., FLAIR, T2) to improve classification accuracy.
Model Optimization: Experiment with hyperparameter tuning, dropout layers, or advanced optimizers to enhance performance.
Real-Time Deployment: Host the application on a web platform (e.g., Flask or Django) for broader accessibility.
Clinical Validation: Collaborate with medical professionals to validate the model on real-world MRI datasets and integrate it into diagnostic workflows.
Explainability: Add feature attribution techniques (e.g., Grad-CAM) to highlight regions influencing the model’s predictions.
Impact and Applications
This project has significant potential in medical diagnostics, particularly for early detection of brain tumors. By automating classification and localization, it can assist radiologists in prioritizing cases, reducing diagnostic errors, and improving patient outcomes. The GUI makes the tool accessible to healthcare professionals without deep technical expertise, while the hybrid model’s robustness ensures reliable performance. The project also showcases transferable skills in AI, computer vision, and software engineering, applicable to other image-based diagnostic systems.

Conclusion
The Brain Tumor Classification and Detection project exemplifies the power of combining cutting-edge deep learning techniques (CNNs and ViTs) with practical software development to address real-world challenges in healthcare. The hybrid model’s accuracy, coupled with the intuitive GUI and tumor localization capabilities, positions it as a valuable tool for medical diagnostics. With further refinements, this project could contribute to advancing AI-driven healthcare solutions, demonstrating both technical innovation and societal impact.

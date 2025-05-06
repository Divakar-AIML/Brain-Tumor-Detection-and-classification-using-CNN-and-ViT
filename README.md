# Brain-Tumor-Detection-and-classification-using-CNN-and-ViT
This brain tumor classification system is a comprehensive deep learning application that combines computer vision and neural networks to automatically detect and classify brain tumors from MRI scans.
1. Core Components
A. Deep Learning Model
Model Loading:

Loads pre-trained hybrid ViT-CNN model ('brain_tumor_classification_model.h5')

Model compiled with Adam optimizer and binary cross-entropy loss

Input shape: (224, 224, 1) grayscale images

Output: 4-class probability distribution (Glioma, Meningioma, Pituitary, No Tumor)

B. Image Processing Pipeline
Preprocessing Function:

Image resizing to 224×224 pixels

Color space conversion (BGR → Grayscale)

Normalization (pixel values scaled to 0-1 range)

Batch dimension addition for model compatibility

Error handling for invalid image paths

Tumor Localization:

Threshold-based segmentation (128 threshold value)

Contour detection using RETR_EXTERNAL mode

Bounding box drawn around largest detected region

Blue rectangle (RGB: 255,0,0) with 2px thickness

C. Classification Logic
Prediction Workflow:

Image preprocessing

Model inference

Class determination (argmax of predictions)

Tumor presence verification (class ≠ "No Tumor")

Bounding box generation conditional on tumor presence

2. GUI Application (Tkinter Implementation)
A. Main Window Configuration
Dimensions: 600×700 pixels

Title: "Brain Tumor Classifier"

Color Scheme: Light gray canvas background

Fonts: Arial (20pt for title, 14pt for results)

B. UI Components
Upload Button:

File dialog restricted to JPG/JPEG/PNG

Image display on canvas (400×400 pixels)

Predict Button:

Triggers classification pipeline

Updates result label and displays annotated image

Canvas:

Displays original and processed images

Maintains aspect ratio through resizing

Result Label:

Dynamic text updates

Clear tumor type indication

C. Event Handling
Image Upload:

File dialog initialization

PIL image conversion and Tkinter PhotoImage adaptation

Canvas update with new image

Prediction Execution:

Validation for uploaded image presence

Error handling through messagebox

Synchronous processing flow

3. Data Flow
User Input:

Image selection via file dialog

Path storage as instance variable

Processing:

OpenCV image loading

Model-compatible tensor creation

GPU-accelerated inference (if available)

Output Generation:

Classification result string

Annotated image with bounding box

RGB conversion for proper display

UI Update:

Text label modification

Canvas image replacement

Memory management of PhotoImage objects

4. Technical Specifications
A. Dependencies
Core Libraries:

TensorFlow 2.x (model loading/inference)

OpenCV (image processing)

Pillow (image display)

NumPy (array operations)

GUI Framework:

Tkinter (native Python UI)

filedialog (file selection)

messagebox (user alerts)

B. File Structure
Model File: brain_tumor_classification_model.h5

Input Images: User-selected via dialog

Output Images: Saved to static/output_images/

C. Performance Considerations
Image Handling:

On-the-fly resizing

Memory-efficient display

Batch processing capability

Model Optimization:

Pre-loaded model weights

Single-image prediction mode

GPU acceleration support

5. Complete Workflow
Initialization:

Model loading and compilation

GUI window creation

Widget placement

User Interaction:

Image selection

Prediction triggering

Result visualization

Background Processing:

Image → Tensor conversion

Neural network inference

Post-processing (bounding box)

Output Display:

Classification result

Visual annotation

Persistent until next operation

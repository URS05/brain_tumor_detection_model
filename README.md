# Brain Tumor Detection System

A professional GUI application for detecting brain tumors using your trained TensorFlow model from Google Teachable Machine.

## Features

🧠 **Tumor Detection**: Detects 4 types of conditions:
- Glioma
- Meningioma  
- Pituitary Tumor
- No Tumor

🎯 **Advanced GUI Features**:
- Drag & Drop support for JPG images
- Batch processing of multiple images
- Real-time progress tracking
- Confidence percentage display
- Professional color scheme and styling
- Tabbed interface (Detection + History)

📊 **Analysis & History**:
- Save results to text files
- Export analysis history to CSV
- Automatic history tracking
- Detailed analysis summaries

## Installation

1. **Install Python** (3.8 or higher recommended)

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your model files are in the correct location**:
   - Model: `C:\drive D\experiment\brain tumor\resources\converted_keras\keras_model.h5`
   - Labels: `C:\drive D\experiment\brain tumor\resources\converted_keras\labels.txt`

## Usage

1. **Run the application**:
   ```bash
   python brain_tumor_gui.py
   ```

2. **Upload Images**:
   - Click "📁 Select Images" to browse for JPG files
   - OR drag & drop JPG files into the designated area

3. **Analyze Images**:
   - Click "🔍 Analyze" to process the selected images
   - View real-time progress and results

4. **Save Results**:
   - Click "💾 Save Results" to save analysis to a text file
   - View history in the "History" tab
   - Export history to CSV for further analysis

## Model Information

Your model can classify brain scans into these categories:
- **Glioma**: A type of brain tumor that begins in glial cells
- **Meningioma**: A tumor that arises from the meninges
- **Pituitary**: A tumor in the pituitary gland
- **No Tumor**: Healthy brain tissue

## Technical Details

- **Input Size**: 224x224 RGB images
- **Model Format**: TensorFlow/Keras .h5 file
- **Preprocessing**: Automatic resizing and normalization
- **Output**: Prediction with confidence percentage

## File Structure

```
working_model/
├── brain_tumor_gui.py          # Main GUI application
├── run.py                      # Simple version (backup)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── brain_tumor_history.json    # Auto-generated history file
└── results/                    # Saved analysis results
```

## Troubleshooting

**Model Loading Issues**:
- Ensure the model file path is correct
- Check that the keras_model.h5 file exists
- Verify the labels.txt file is properly formatted

**Image Processing Issues**:
- Only JPG format is supported
- Images are automatically resized to 224x224
- Ensure images are valid brain scan images

**Dependencies Issues**:
- Run: `pip install --upgrade tensorflow keras numpy pillow pandas tkinterdnd2`
- For GPU support: `pip install tensorflow-gpu`

## Support

The application includes:
- Error handling with user-friendly messages
- Progress tracking for batch processing
- Automatic history saving
- Professional styling and layout

For issues, check the console output for detailed error messages.

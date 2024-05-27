# Generative AI: CelebA Dataset

This project is a Python application designed to work with the CelebA dataset. It is intended to be run in a Google Colab environment.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The project requires the following Python packages:

- os
- cv2
- numpy
- zipfile

You can install these packages using pip:

```bash
pip install opencv-python numpy
```

The `os` and `zipfile` modules are part of the Python Standard Library, so you don't need to install them separately.

## Dataset

The project uses the CelebA dataset. Here's how you can download it and upload it to Google Drive:

1. Go to the CelebA dataset website: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Scroll down to the "Download" section.
3. You will find several download options. Choose the one that best suits your needs.
4. After downloading the dataset, you should have a zip file.
5. Upload this zip file to your Google Drive.

## Running the Script in Google Colab

To run the script in Google Colab, you'll need to mount your Google Drive to access the CelebA dataset zip file. Here's how you can do it:

1. Open the notebook in Google Colab.
2. In a code cell, run the following command:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Follow the instructions to authorize Google Colab to access your Google Drive.
Once your drive is mounted, you can access files in your Google `Drive under /content/drive/My Drive/.`

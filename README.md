# Stellar Classification System

This project develops a machine learning model to classify stars based on photometric data from the **Sloan Digital Sky Survey (SDSS)**. Using the `u`, `g`, `r`, `i`, and `z` photometric filter bands, the model predicts the class of astronomical objects.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview
The **Stellar Classification System** is built to classify celestial objects (specifically stars) using their photometric data. The model utilizes data from the **SDSS** photometric survey, including ultraviolet, green, red, and infrared magnitudes, to classify stars into different categories.

The current version of the project employs the **K-Nearest Neighbors (KNN)** algorithm, achieving a classification accuracy of **88%**.

## Project Structure
```
├── classification.py    # Main script to train and evaluate the model
├── requirements.txt     # List of dependencies to install
├── README.md            # Project overview (this file)
└── Skyserver_data.csv   # SDSS photometric data used for training
```

## Dataset
The dataset used in this project is obtained from the **Sloan Digital Sky Survey (SDSS)** and includes the following columns:

- **objid**: Object ID
- **ra**: Right Ascension (RA) of the object
- **dec**: Declination (Dec) of the object
- **u**: Ultraviolet magnitude (354 nm)
- **g**: Green magnitude (477 nm)
- **r**: Red magnitude (623 nm)
- **i**: Near-infrared magnitude (763 nm)
- **z**: Far-infrared magnitude (913 nm)
- **class**: The class of the object (e.g., STAR, GALAXY)

The dataset is stored as `Skyserver_data.csv`.

## Installation
To run the project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Stellar_Classification.git
    cd Stellar_Classification
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv env
    source env/bin/activate  # For Windows, use env\Scripts\activate
    ```

3. **Install dependencies**:
    Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Once dependencies are installed, run the model training script by executing:

```bash
python classification.py
```

The script will:
- Load the dataset
- Preprocess the data
- Train the KNN model using the SDSS photometric data
- Output the accuracy and the best parameters

### Example output:
```
Accuracy: 88.00%
Best Parameters: {'n_neighbors': 1}
```

## Model
The current model is a **K-Nearest Neighbors (KNN)** classifier. The features used for classification are the magnitudes from the following photometric bands: `u`, `g`, `r`, `i`, `z`.

The optimal number of neighbors (`n_neighbors`) was determined using **GridSearchCV**, and the best result was achieved with `n_neighbors=1`.

## Results
The model achieves an accuracy of **88%** on the test set. The KNN classifier works well for this dataset, but additional models or data preprocessing techniques may further improve accuracy.

## Future Improvements
Some potential improvements to the project include:
- Experimenting with other classification algorithms (e.g., SVM, Random Forest, XGBoost)
- Further data cleaning and preprocessing (e.g., handling missing values)
- Using additional features from the SDSS dataset, such as redshift or other spectral properties
- Implementing cross-validation and further hyperparameter tuning

## Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request. If you find any issues or have ideas for improvements, please open an issue in the repository.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



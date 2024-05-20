Here's the corrected version:

---

# End-to-End Machine Learning Project with Python

The project aims to predict `student math performance` based on personal, family, and social information. 

The main idea of this project is to implement an *end-to-end machine learning* project in a structured format. Using the `Object-Oriented programming approach`, I've structured the project into different sections. The source folder contains the main code with components and pipeline sections. Logs are stored under the logs folder, containing information, warnings, or errors.

To run the project, follow these steps:

1. Clone the repository
```bash
git clone https://github.com/Vignesh142/ML_Project-OOPs
```

2. Change directory
```bash
cd ML_Project-OOPs
```

3. Install the required libraries
```bash
pip install -r requirements.txt
```

4. Run the app.py file
```bash
python application.py
```
---

The project is divided into the following sections:

### 1. Data Collection
- Data is collected from `Kaggle` datasets.
- Data is in CSV format and includes students' personal, family, and social information.

### 2. Data Preprocessing
- Data preprocessing is performed using the pandas library.
- Cleaning and handling missing values are conducted.
- Data is normalized, encoded, and scaled.

### 3. Data Splitting
- Data is split into training and testing datasets.
- Features and labels are separated.

### 4. Model Building
- Models are built using the scikit-learn library.
- Models are trained on the training dataset and evaluated on the testing dataset.
- The model predicts the student's math score out of 100.
- Various regression techniques are utilized including `Linear Regression, Decision Tree Regression, Random Forest Regression, Gradient Boosting Regression, AdaBoost Regression, XGBoost Regression, and K-Nearest Neighbors Regression.`
- Hyperparameter tuning is performed using `GridSearchCV`.
- The best model is selected based on evaluation metrics.
- Models are saved using the joblib library.

### 5. Model Evaluation
- Models are evaluated using metrics such as `Mean Squared Error, Mean Absolute Error, and R2 Score`.
- Evaluation metrics are calculated using the scikit-learn library.

### 6. Model Deployment
- Models are deployed using the `Flask web framework`.
- Deployment is done on the local server.

### 7. Model Testing
- Models are tested using the `ThunderClient API` testing in VSCode.

### 8. Model Logging
- Logs are stored under the `logs` folder.
- Logs contain information, warnings, and errors regarding the model.

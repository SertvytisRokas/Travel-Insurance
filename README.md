# DS.v2.5.3.1.5

# Travel Insurance predictions

A Tour & Travels company is offering a travel insurance package to their customers, which now includes Covid Cover. This insurance was offered to some customers in 2019, and the provided data has been extracted from the performance and sales of the package during that period.

## Objectives
- Identify the most accurate model based on business needs.
- Provide business recommendations on targeting individuals for the special Covid Cover travel insurance plan.

## Methodology
### Data Preprocessing
Handled missing values, encoded categorical features, and scaled numerical features.

### Model Selection
Tested 5 types of machine learning models:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### Model Training
Split data into training and testing sets, tuned hyperparameters, and evaluated models using accuracy, precision, recall, F1-score, and AUC-ROC curves.

### Ensemble Learning
Combined predictions from multiple models using a Voting Classifier to improve accuracy.

## Results & Recommendations
Evaluated models and provided actionable business recommendations to target potential customers for the Covid Cover insurance plan.

## How to Run
1. Clone the repository.
2. Install required packages from requirements.txt.
3. Run the Jupyter notebook to explore data analysis and model training steps.

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
jupyter notebook Travel Insurance predictions.ipynb

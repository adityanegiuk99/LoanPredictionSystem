# Loan Prediction using Support Vector Machine

This project aims to build a predictive model using a Support Vector Machine (SVM) to determine the eligibility of individuals for a loan based on various factors. The goal is to assist financial institutions in making informed decisions about loan applications.
 
## Dataset Description

The dataset used in this project is sourced from [mention source if known, otherwise state 'an unspecified source']. It contains information about various attributes of individuals applying for a loan and their corresponding loan status.

The dataset consists of the following key features:

*   **Loan\_ID**: Unique identifier for each loan application.
*   **Gender**: Applicant's gender (Male/Female).
*   **Married**: Applicant's marital status (Yes/No).
*   **Dependents**: Number of dependents the applicant has.
*   **Education**: Applicant's education level (Graduate/Not Graduate).
*   **Self\_Employed**: Whether the applicant is self-employed (Yes/No).
*   **ApplicantIncome**: Applicant's monthly income.
*   **CoapplicantIncome**: Co-applicant's monthly income.
*   **LoanAmount**: The requested loan amount.
*   **Loan\_Amount\_Term**: The term of the loan in months.
*   **Credit\_History**: Applicant's credit history (1.0 for good, 0.0 for bad).
*   **Property\_Area**: The area of the property (Rural/Semiurban/Urban).

The target variable for this prediction model is:

*   **Loan\_Status**: The status of the loan application (Y for approved, N for rejected). This is converted to numerical values (1 for approved, 0 for rejected) for model training.


## Data Preprocessing

Data preprocessing is a crucial step to prepare the dataset for the machine learning model. This involves handling missing values and converting categorical features into a numerical format that the model can understand.

1.  **Handling Missing Values**: Initially, the presence of missing values in various columns was identified. To address this, rows containing any missing values were removed from the dataset. This decision was made based on the executed code which showed that dropping rows with missing values resulted in a dataset with no null entries.

2.  **Label Encoding the Target Variable**: The target variable 'Loan_Status', which is categorical ('Y' for approved and 'N' for rejected), was converted into numerical form. 'N' was replaced with `0` and 'Y' was replaced with `1`.

3.  **Handling 'Dependents' Column**: The 'Dependents' column contained the value '3+' which was replaced with the numerical value `4` to represent the number of dependents.

4.  **Encoding Categorical Features**: Other categorical features in the dataset, such as 'Married', 'Gender', 'Self_Employed', 'Property_Area', and 'Education', were converted into numerical representations. 'Married' ('No' to 0, 'Yes' to 1), 'Gender' ('Male' to 1, 'Female' to 0), 'Self_Employed' ('Yes' to 1, 'No' to 0), 'Property_Area' ('Rural' to 0, 'Semiurban' to 1, 'Urban' to 2), and 'Education' ('Graduate' to 1, 'Not Graduate' to 0) were replaced with their corresponding numerical values.


   ## Data Visualization

Exploratory data analysis was performed to visualize the relationship between key features and the loan status.

1.  **Education vs. Loan Status**: The count plot for 'Education' and 'Loan_Status' shows the distribution of loan approvals and rejections based on the applicant's education level. It indicates that a higher number of graduates were approved for loans compared to non-graduates, although the proportion of approvals within each education group can be further analyzed.

2.  **Married vs. Loan Status**: The count plot for 'Married' and 'Loan_Status' illustrates how marital status relates to loan approval. The plot suggests that married individuals have a higher count of both loan approvals and rejections compared to non-married individuals, and a larger proportion of married applicants were approved for loans.



## Model Training

A Support Vector Machine (SVM) model with a linear kernel was used for predicting loan eligibility. The model was trained on the preprocessed training data, which includes the features in `X_train` and the corresponding loan statuses in `Y_train`. The training process involves the SVM algorithm finding the optimal hyperplane that separates the data points into different classes (loan approved or not approved) based on the provided features.


## Model Evaluation

The performance of the Support Vector Machine model was evaluated using the accuracy score, which measures the proportion of correctly predicted instances.

The accuracy score obtained on the training data is: `{{train_data_accuracy}}`.
The accuracy score obtained on the test data is: `{{test_data_accuracy}}`.

These scores indicate how well the model performed on the data it was trained on (training data) and on unseen data (test data). A good model should have high accuracy on both, with the test accuracy being a more reliable indicator of the model's generalization ability. In this case, the model shows good performance on both training and test sets, suggesting it has learned the patterns in the data effectively and can generalize well to new data.


## Predictive System

This section describes how to use the trained Support Vector Machine model to predict the loan eligibility for a new data instance.

To make a prediction for a new applicant, you need to provide their information in the form of a tuple or list of numerical values. The order of these values must correspond to the features used for training, excluding 'Loan_ID' and 'Loan_Status'. Based on the data preprocessing steps, the required features and their expected numerical format are as follows:

1.  **Gender**: 1 for Male, 0 for Female
2.  **Married**: 1 for Yes, 0 for No
3.  **Dependents**: Numerical value (0, 1, 2, or 4 for '3+')
4.  **Education**: 1 for Graduate, 0 for Not Graduate
5.  **Self_Employed**: 1 for Yes, 0 for No
6.  **ApplicantIncome**: Numerical value representing the applicant's income.
7.  **CoapplicantIncome**: Numerical value representing the co-applicant's income.
8.  **LoanAmount**: Numerical value representing the loan amount.
9.  **Loan_Amount_Term**: Numerical value representing the loan term in months.
10. **Credit_History**: 1.0 for good credit history, 0.0 for bad credit history.
11. **Property_Area**: 0 for Rural, 1 for Semiurban, 2 for Urban

Here are the steps to make a prediction:

1.  **Prepare Input Data**: Create a tuple or list containing the numerical values for the new applicant following the order of features listed above.
2.  **Convert to NumPy Array**: Convert the input data (tuple/list) into a NumPy array.
3.  **Reshape the Array**: Reshape the NumPy array to indicate that it's a single instance being predicted.
4.  **Make Prediction**: Use the trained `classifier.predict()` method with the reshaped input data.
5.  **Interpret Result**: The prediction result will be a NumPy array containing a single value: `1` indicates the person is eligible for the loan, and `0` indicates the person is not eligible.

Example:
```python
input_data = (1, 1, 1, 1, 0, 4583, 1508, 128, 360, 1, 0) # Example input data

# change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print("The person is not eligible for loan")
else:
    print("The person is eligible for loan")
```


## How to Run the Code

To run this loan prediction project, follow these steps:

1.  **Clone the Repository (if applicable):** If the code is hosted in a repository, clone it to your local machine.
    ```bash
    git clone <repository_url>
    ```
2.  **Install Dependencies:** The project requires the following Python libraries:
    *   `pandas`
    *   `numpy`
    *   `scikit-learn`
    *   `seaborn`

    You can install these dependencies using pip:
    ```bash
    pip install pandas numpy scikit-learn seaborn
    ```
3.  **Obtain the Dataset:** Ensure the dataset file named `loan_dataset.csv` is in the same directory as the Jupyter Notebook file, or update the file path in the code cell that loads the data (`pd.read_csv('/content/loan_dataset.csv')`) to the correct location of your dataset file.
4.  **Run the Jupyter Notebook:**
    *   Open your terminal or command prompt.
    *   Navigate to the directory where you saved the notebook and the dataset.
    *   Start the Jupyter Notebook server by running the command:
        ```bash
        jupyter notebook
        ```
    *   Your web browser should open with the Jupyter Notebook dashboard.
    *   Click on the notebook file (`.ipynb`) to open it.
    *   Run the cells sequentially by clicking on each cell and pressing `Shift + Enter`, or by going to the "Cell" menu and selecting "Run All".




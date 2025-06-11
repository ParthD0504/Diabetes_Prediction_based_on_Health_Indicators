This repository contains a complete machine learning pipeline to predict the onset of diabetes based on individual health indicators. It includes data loading, exploratory data analysis (EDA), preprocessing, model development, evaluation, and hyperparameter tuning using a variety of algorithms.

## Table of Contents

* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Data](#data)
* [Preprocessing](#preprocessing)
* [Models](#models)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/diabetes-predictor.git
   cd diabetes-predictor
   ```
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your dataset CSV file in the `data/` directory (e.g., `data/diabetes_data.csv`).
2. Run the data loading and EDA notebook:

   ```bash
   jupyter notebook notebooks/Diabetes_Classifier_EDA.ipynb
   ```
3. Execute preprocessing and split the data by running:

   ```bash
   python -c "from src.load_data import load_dataset; from src.preprocess import Preprocess; df = load_dataset('data/diabetes_data.csv'); Preprocess(target_col='Outcome', numeric_cols=[...]).fit_transform(df)"
   ```
4. Explore model notebooks in sequence:

   * `notebooks/discrete_naive_bayes.ipynb`
   * `notebooks/logistic_regression.ipynb`
   * `notebooks/support_vector_machine.ipynb`
   * `notebooks/svm_rff_tuned.ipynb`
   * `notebooks/xg_boost_final.ipynb`

## Project Structure

```
├── data/                         # Raw and processed data files
├── notebooks/                    # Jupyter notebooks for EDA and modeling
│   ├── Diabetes_Classifier_EDA.ipynb
│   ├── discrete_naive_bayes.ipynb
│   ├── logistic_regression.ipynb
│   ├── support_vector_machine.ipynb
│   ├── svm_rff_tuned.ipynb
│   └── xg_boost_final.ipynb
├── src/                          # Source code modules
│   ├── load_data.py              # Data loading utilityfileciteturn0file0
│   ├── preprocess.py             # Preprocessing class for scaling and splittingfileciteturn0file1
│   └── svm_rff.py                # Custom SVM with Random Fourier Featuresfileciteturn0file2
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview and instructions
```

## Data

* **Source:** Place a CSV file containing health indicators and an `Outcome` column (0=no diabetes, 1=diabetes).
* **Features:** Age, BMI, Glucose, BloodPressure, SkinThickness, Insulin, DiabetesPedigreeFunction, etc.
* **Target:** `Outcome` (binary label).

## Preprocessing

* **Robust Scaling:** Numeric features are robust-scaled using median and interquartile range to reduce outlier impact.
* **Ordinal Inversion:** Select ordinal columns (e.g., general health ratings) can be inverted for correct ordering.
* **Train/Test Split:** Stratified split (80/20) to preserve class balance.

## Models

1. **Discrete Naive Bayes** — Baseline probabilistic classifier.
2. **Logistic Regression** — Linear model with L2 regularization.
3. **Support Vector Machine** — Kernel-based classifier using scikit-learn.
4. **Custom SVM with Random Fourier Features** — Approximates Gaussian kernel for scalability.
5. **XGBoost** — Gradient-boosted decision trees for high performance.

Each notebook covers training, hyperparameter tuning, and evaluation (accuracy, precision, recall, ROC-AUC).

## Results

Performance metrics and visualizations are provided in each notebook. In our experiments, tree-based models (XGBoost) typically achieve the highest ROC-AUC, while linear models provide interpretable baselines.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Parth Deshmukh — [deshmukh.par@northeastern.edu](mailto:deshmukh.par@northeastern.edu)

Feel free to reach out with questions or feedback!")}


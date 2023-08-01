# Lung_Cancer_DataAnalysis

DataSet - https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link

# 1. Feature Importance and Predictive Modeling : Feature importance analysis for predicting the level of lung cancer using a RandomForestClassifier. Here is a summary of the steps:

Data Preprocessing: The dataset is loaded from a CSV file, and the features and target variable are separated. The 'index', 'Patient Id', and 'Level' columns are dropped from the features (X), and the 'Level' column is used as the target variable (y).

Feature Importance Analysis: A RandomForestClassifier is initialized and trained on the entire dataset (X and y) to calculate the feature importances. The feature importances are then stored in a DataFrame and sorted in descending order to identify which features have the most significant impact on the prediction.

Model Selection and Splitting: The dataset is split into training and testing sets using train_test_split function from sklearn.model_selection. 80% of the data is used for training (x_train and y_train), and 20% is used for testing (X_test and y_test).

Model Training and Evaluation: Another instance of the RandomForestClassifier is trained on the training data (x_train and y_train), and predictions are made on the testing data (X_test). The accuracy of the model is evaluated using the accuracy_score function from sklearn.metrics, and a classification report and confusion matrix are printed to further evaluate the model's performance.

Interpretation of Feature Importance: Finally, a bar plot is created using matplotlib to visualize the feature importances. This plot helps in identifying which features have the most significant impact on predicting the level of lung cancer.

# 2. Gender Differences in Lung Cancer Risk: Analyzes and visualizes the distribution of lung cancer levels based on gender using a dataset of cancer patient data.

Load the Dataset: The dataset is loaded from a CSV file named 'cancer patient data sets.csv' into a DataFrame called data1.

Data Preparation: The data is split into two DataFrames, male_data and female_data, based on the 'Gender' column. male_data contains rows where the 'Gender' value is 1 (representing males), and female_data contains rows where the 'Gender' value is 2 (representing females).

Distribution of Lung Cancer Levels: The number of occurrences of each lung cancer level for both males and females is calculated and stored in the variables male_level_distribution and female_level_distribution, respectively.

Plot Distribution of Lung Cancer Levels for Each Gender: Two pie charts are created side by side using matplotlib to visualize the distribution of lung cancer levels for males and females. The pie charts show the percentage distribution of different lung cancer levels (e.g., Stage 1, Stage 2, etc.) for each gender.

Save and Display the Plot: The pie charts are saved as 'distribution of lung cancer levels gender.png' and displayed using plt.show().

# 3. Lifestyle and Lung Cancer: analysis of the relationship between three lifestyle factors (Smoking, Alcohol use, Obesity) and the risk of lung cancer using a dataset of cancer patient data.

Import seaborn library: The code imports the seaborn library as sns. Seaborn is a Python data visualization library based on matplotlib that provides high-level interface for creating informative and attractive statistical graphics.

Load the Dataset: The dataset is loaded from a CSV file named 'cancer patient data sets.csv' into a DataFrame called data3.

Data Preparation: Three lifestyle factors (Smoking, Alcohol use, Obesity) and the target variable (Level, representing lung cancer stage) are defined.

Analyze Lifestyle Factors and Lung Cancer: The code creates a single row of subplots (side by side) to visualize the relationship between each lifestyle factor and the risk of lung cancer. For each lifestyle factor, a box plot is created using seaborn's boxplot function. The box plot represents the distribution of each lifestyle factor's values for different lung cancer stages. The x-axis represents the lung cancer stage, and the y-axis represents the count or frequency of the lifestyle factor. The title of each subplot indicates the lifestyle factor being analyzed.

Save and Display the Plot: The subplots are saved as 'lifestyle_factors_and_lung_cancer_risk.png' and displayed using plt.show().

# 4. Association of Environmental Factors with Lung Cancer: analysis of the relationship between three environmental factors (Air Pollution, Dust Allergy, Occupational Hazards) and the prevalence of lung cancer using a dataset of cancer patient data.

Load the Dataset: The dataset is loaded from a CSV file named 'cancer patient data sets.csv' into a DataFrame called data4.

Data Preparation: Three environmental factors (Air Pollution, Dust Allergy, Occupational Hazards) and the target variable (Level, representing lung cancer stage) are defined.

Analyze Environmental Factors and Lung Cancer Prevalence: The code creates a single row of subplots (side by side) to visualize the relationship between each environmental factor and the prevalence of lung cancer. For each environmental factor, a count plot is created using seaborn's countplot function. The count plot shows the count or frequency of each environmental factor for different lung cancer stages. The x-axis represents the environmental factor, and the hue (different colored bars) represents the lung cancer stage. The title of each subplot indicates the environmental factor being analyzed.

Save and Display the Plot: The subplots are saved as 'environmental_factors_and_lung_cancer_prevalence.png' and displayed using plt.show().

# 5. Comorbidity Analysis: analysis of the relationship between chronic lung diseases and the risk of lung cancer using a dataset of cancer patient data.

Load the Dataset: The dataset is loaded from a CSV file named 'cancer patient data sets.csv' into a DataFrame called data5.

Data Preparation: The chronic lung disease (chronic Lung Disease) and the target variable (Level, representing lung cancer stage) are defined.

Analyze Comorbidity and Lung Cancer Risk: The code creates a single plot to visualize the relationship between chronic lung diseases and the risk of lung cancer. A count plot is created using seaborn's countplot function. The count plot shows the count or frequency of chronic lung diseases for different lung cancer stages. The x-axis represents the presence or absence of chronic lung disease, and the hue (different colored bars) represents the lung cancer stage. The title of the plot indicates the comorbidity being analyzed.

Save and Display the Plot: The plot is saved as 'comorbidity_and_lung_cancer_risk.png' and displayed using plt.show().

# 6. Symptom Clustering and Disease Progression : clustering analysis using the k-means algorithm on cancer patient data to identify clusters of patients based on their symptoms. It then analyzes the disease progression and response to treatment for each cluster.

Import Necessary Libraries: The code imports the required libraries, including KMeans for clustering and StandardScaler for feature scaling.

Load the Dataset: The dataset is loaded from a CSV file named 'cancer patient data sets.csv' into a DataFrame called data6.

Data Preparation: The symptoms data is extracted from the DataFrame data6 by selecting columns from the 5th column (index 4) to the second-to-last column.

Feature Scaling: The symptoms data is standardized using StandardScaler to bring all the features to a common scale.

Apply Clustering Algorithm (k-means): The k-means algorithm is applied to the scaled symptoms data with the number of clusters (num_cluster) set to 4. The algorithm assigns each patient to one of the four clusters based on their symptoms.

Analyze Cluster and Disease Progression: The cluster assignments are added to the DataFrame data6. The code then creates a count plot to visualize the distribution of patients in each cluster and their lung cancer levels. The x-axis represents the clusters, and the hue (different colored bars) represents the lung cancer levels. This plot provides insights into the relationship between clusters and lung cancer stages.

Analyze Disease Progression and Response to Treatment for Each Cluster: The code calculates the percentage of patients in each cluster belonging to different lung cancer levels and prints the results as cluster_summary. This summary helps understand how each cluster is associated with different disease progressions and responses to treatment.

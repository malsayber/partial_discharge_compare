


1. *Load data* from a standard dataset (e.g., Iris dataset).
2. *Process data*, including feature scaling and train-test splitting.
3. *Evaluate different traditional machine learning models using multiple hyperparameter settings.
4. *Analyze different feature subsets*, systematically testing various feature combinations to determine the best-performing ones.
   4a *Use different feature selection packages like featurewiz, featuretools, and mljar-supervised*
5. *Use grid search or similar techniques* to find the optimal hyperparameters for each model.
6. *Output comparative results*, summarizing model performance (accuracy and classification report) across different feature combinations and parameter settings, and for different packages.
7. *Ensure modularity*, separating functionalities into distinct files (e.g., data_loader.py, data_processor.py, model_runner.py, parameter_tuner.py, feature_selector.py, and main.py) for easy debugging and reusability."
8. have the option to use either multiple cpu core or single thread for training the model based on given keyword argument
9. have the option to use either multiple gpu or single gpu for training the model based on given keyword argument, but need to check if the gpu is available or not
10. for each model training, always include the Checkpointing and resuming training from the last checkpoint. 
11. there is logger for each model training, and the log will be saved in the log file
12. there is master log that detail out what is the model that is being trained, the time taken for the training, the accuracy of the model, the classification report of the model, the feature selection method used, the hyperparameter tuning.
13. you need suggest the best model based on the different metric report
14. you need to have the option to save the model in the pickle file
15. you need to have the option to save the model in the onnx file

# featurewiz

Automatically select the most relevant features without specifying a number 🚀 Fast and user-friendly, perfect for data scientists at all levels 🎯 Provides a built-in categorical-to-numeric encoder 📚 Well-documented with plenty of examples 📝 Actively maintained and regularly updated

https://github.com/AutoViML/featurewiz


# MLJAR Supervised

The mljar-supervised is an Automated Machine Learning Python package that works with tabular data. It is designed to save time for a data scientist. It abstracts the common way to preprocess the data, construct the machine learning models, and perform hyper-parameters tuning to find the best model 🏆. It is no black box, as you can see exactly how the ML pipeline is constructed (with a detailed Markdown report for each ML model).

https://github.com/mljar/mljar-supervised

# Featuretools
Featuretools is a framework to perform automated feature engineering. It excels at transforming temporal and relational datasets into feature matrices for machine learning.

https://featuretools.alteryx.com/en/stable/#


*"Generate a modular Python codebase that compares different conventional machine learning models without requiring a GPU. The code should be structured into multiple .py files for easy debugging and maintainability. It should perform the following tasks:* 
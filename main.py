import streamlit as st
import os
from sklearn.datasets import fetch_california_housing
import base64
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC, SVR
import matplotlib.pyplot as plt
from DataCleaningFunctions import completness_ratio, data_val_erroneous, data_val_duplicates
from groq import Groq
import seaborn as sns
import time
import xgboost
import requests
client = Groq(api_key= "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")


def get_video_as_base64(file_path):
    with open(file_path, "rb") as video_file:
        video_bytes = video_file.read()
    return base64.b64encode(video_bytes).decode()

def identify_variable_types(df):
    """
    Identify continuous and discrete variables in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame to analyze.

    Returns:
    dict: A dictionary with 'continuous' and 'discrete' keys containing lists of variable names.
    """
    variable_types = {
        'continuous': [],
        'discrete': []
    }
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            unique_values = df[column].nunique()
            if unique_values > 20:  # Threshold to differentiate continuous and discrete
                variable_types['continuous'].append(column)
            else:
                variable_types['discrete'].append(column)
        else:
            variable_types['discrete'].append(column)  # Non-numeric types are treated as discrete
    
    return variable_types

def typewriter(text: str, speed: int):
                tokens = text.split()
                container = st.empty()
                for index in range(len(tokens) + 1):
                    curr_full_text = " ".join(tokens[:index])
                    container.markdown(curr_full_text)
                    time.sleep(1 / speed)

speed = 20

def llama31_70b(prompt, temperature=0.0, input_print=True):
  chat_completion = client.chat.completions.create(
      messages=[
          {
              "role": "user",
              "content": prompt,
          }
      ],
      model="llama3-70b-8192",
      temperature=temperature,
  )

  return (chat_completion.choices[0].message.content)

st.set_page_config(page_title="Data Intelligence Hub", page_icon=":rocket", layout="wide")


# st.title("Welcome to The Data Intelligence Hub")


# Set a background image using CSS
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://w0.peakpx.com/wallpaper/855/509/HD-wallpaper-simple-background-edit-and-simple-design.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}

.lottie-container {
    background: transparent !important;  /* Ensures Lottie animation has no background */
    display: flex;
    justify-content: center;
    align-items: center;
    height: auto;  /* Adjust as needed */
    width: auto;  /* Adjust as needed */
    margin-top: 20px; /* Optional: Adjust margin for positioning */
}
</style>
'''

# Apply CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Welcome to The Data Intelligence Hub 🚀")
typewriter("Unlock the potential of your data with our self-service analytics tool, designed for professionals at every level. Seamlessly perform advanced analyses such as regression, clustering, and predictive modeling—no coding required. Empower your decision-making with actionable insights at your fingertips.", speed=speed)
st.markdown("---")

with st.spinner('Setting up your environment...'):
    video_base64 = get_video_as_base64("Data App Demo.mp4")
    
    # Create the HTML for the video with custom size
    video_html = f"""
    <video width="600" height="400" controls autoplay muted loop>
  <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
  Your browser does not support the video tag.
            </video>
"""
    
    # Divide layout into two columns
    col2, col1 = st.columns([1, 1])  # Adjust the width ratio if needed
    
    # Render the video in the first column
    with col1:
        st.markdown(video_html, unsafe_allow_html=True)
    
    # Render the text in the second column
    with col2:
        st.subheader("How Can the Data Intelligence Hub Help You?")
        st.write("""
        As your Data Science Agent, the Data Intelligence Hub platform enables you to develop robust data science solutions effortlessly, without any coding. It provides intelligent recommendations throughout the process and interprets results for you, ensuring a seamless experience. Here’s what you can do:
    
        - Ensure your data meets quality and completeness standards
        - Build statistical regression models with ease
        - Perform clustering analysis to discover hidden patterns
        - Develop predictive machine learning models
        - Utilize dimensionality reduction techniques (Coming soon...)
        - Conduct hypothesis testing to validate insights (Coming soon...)
        - And much more!
        """)
    
st.markdown("---")
# File uploader for user's dataset
uploaded_file = st.file_uploader("Upload Your Dataset", type=["csv", "xlxs", "xls"])

# Improved text for users who don't have a dataset
st.write("Don't have a dataset? No problem! You can explore the app using one of our sample datasets below.")

# Selection box for built-in datasets
selected_dataset = st.selectbox(
    "Choose a sample dataset to get started:",
    ["No Selection", "Student Performance Dataset", "Housing Market Dataset"]
)

# Visual touch to separate sections
st.markdown("---")


if uploaded_file is not None or selected_dataset != "No Selection":
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        if selected_dataset == "Housing Market Dataset":
            df = fetch_california_housing()
            df = pd.DataFrame(df.data, columns=df.feature_names)
        elif selected_dataset == "Student Performance Dataset":
            df = pd.read_csv("Student_performance_data.csv")
    st.subheader("Data Preview")
    st.write(df.head(5))

    data_completness = completness_ratio(df)
    data_valid = data_val_erroneous(df)
    data_dup = data_val_duplicates(df)
    template =  """You role is to provide a nice and concise summary of the data quality and completness.
                You will do this by taking in the given information and synthesizing the data quality and completness to the user. For the data completness, concisely list all the variables and 
                their complteness ration as well"""
    data_check_prompt = f"""Data completness: {data_completness}, 
                Data Type Validation for each column: {data_valid}, 
                Data Duplicate Validation: {data_dup}"""
    
    prompt_text = template  + "'''" + data_check_prompt  + "'''" 

    output = llama31_70b(prompt_text)
    st.write("**Data Completness & Quality**")
    st.write(output)


    st.subheader("Let's Get Started")
    analysis_modes = ["Regression Analysis", "Clustering Analysis", "Predictive Analysis", "Data Vizualisation", "Text Analysis", "Dimensionality Reduction", "Hypothesis Testing"]
    analysis_mode = st.selectbox("Select Data Analysis Mode", analysis_modes)

    if analysis_mode == "Regression Analysis":
        st.subheader("Regression Analysis")
        df_columns = df.columns.tolist()
        y = st.selectbox("Select the depedent variable", df_columns)
        available_x_columns = [col for col in df_columns if col != y]
        X = st.multiselect("Select the indepedent variable", available_x_columns)
        regression_models = ["OLS Regression", "Ridge Regression", "LASSO Regression", "Elastic Net Regression"]
        # if len(X) > 1:
        #     regression_model = st.selectbox("Select regression model", regression_models[1:])
        # else:
        regression_model = st.selectbox("Select regression model", regression_models)
        if regression_model == "OLS Regression":
            if X:
                X = df[X]
                y = df[y]
                model = sm.OLS(y, X)
                results = model.fit()
                st.write(results.summary())
                result_summary = str(results.summary())
                template = "You will summarize and interpret all the important results from the OLS regression concisely while covering all the important details."
                prompt_text = template  + "'''" + result_summary  + "'''" 
                output = llama31_70b(prompt_text)
                st.write(output)
        elif regression_model == "Ridge Regression":
            if X:
                X = df[X]
                y = df[y]
                model = sm.OLS(y, X).fit_regularized(alpha=1.0, L1_wt=0.0)
                # results = model.fit()
                # st.write(model.params)
                coefficients = pd.DataFrame(model.params, index=X.columns, columns=['Coefficient'])
                st.write(coefficients.style.background_gradient(cmap='coolwarm').format(precision=4))
                y_pred = np.dot(X, model.params)
                ss_total = np.sum((y - np.mean(y)) ** 2)
                ss_residual = np.sum((y - y_pred) ** 2)
                r_squared = 1 - (ss_residual / ss_total)
                st.write(f"**R-squared**: {r_squared:.4f}")


                # Visualize the coefficients using a bar plot
        elif regression_model == "LASSO Regression":
            if X:
                X = df[X]
                y = df[y]
                model = sm.OLS(y, X).fit_regularized(alpha=1.0, L1_wt=1.0)
                # results = model.fit()
                # st.write(model.params) 
                coefficients = pd.DataFrame(model.params, index=X.columns, columns=['Coefficient'])
                st.write(coefficients.style.background_gradient(cmap='coolwarm').format(precision=4))
                y_pred = np.dot(X, model.params)
                ss_total = np.sum((y - np.mean(y)) ** 2)
                ss_residual = np.sum((y - y_pred) ** 2)
                r_squared = 1 - (ss_residual / ss_total)
                st.write(f"**R-squared**: {r_squared:.4f}")

        elif regression_model == "Elastic Net Regression":
            if X:
                X = df[X]
                y = df[y]
                model = sm.OLS(y, X).fit_regularized(alpha=0.5, L1_wt=0.5)
                # results = model.fit()
                # st.write(model.params) 
                coefficients = pd.DataFrame(model.params, index=X.columns, columns=['Coefficient'])
                st.write(coefficients.style.background_gradient(cmap='coolwarm').format(precision=4))
                y_pred = np.dot(X, model.params)
                ss_total = np.sum((y - np.mean(y)) ** 2)
                ss_residual = np.sum((y - y_pred) ** 2)
                r_squared = 1 - (ss_residual / ss_total)
                st.write(f"**R-squared**: {r_squared:.4f}")     

    elif analysis_mode == "Clustering Analysis":
        st.subheader("Clustering Analysis")
        clustering_method = st.selectbox("Select the clustering method", ["K Means", "Hierarchical Clustering", "DBSCAN"])
        if clustering_method == "K Means":
            df_columns = df.columns.tolist()
            cluster_features = st.multiselect("Select which features to be clustered on", df_columns)
            unique_id = st.selectbox("If present in the data select the variable that can be used to uniquely identify observation e.g.CustomerID, PurchaseOrderID", 
                                 df_columns, 
                                 index=0, 
                                 help="Leave empty if no unique ID is available.")
            if len(cluster_features) == 1:
                X = df[cluster_features]
                # X = X.apply(pd.to_numeric, errors='coerce')
                # X = X.dropna()
                # X = X.select_dtypes(include=[np.number])
                k = st.number_input("Select the number of clusters k", min_value=1, step=1)
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X)
                centroids = kmeans.cluster_centers_
                df_centroids = pd.DataFrame({
                "Cluster": range(1, len(centroids) + 1),  # Cluster numbers (1-based indexing)
                "Centroid": centroids.flatten()           # Centroid values for each cluster
                })
                df_cluster_assignment = pd.DataFrame({
                "Cluster": kmeans.labels_,               # Cluster labels for each point
                "Centroid": kmeans.cluster_centers_[kmeans.labels_, 0], 
                "Observation": X[cluster_features].values.flatten()  # Original values
                }, index=X.index)
                df_merged = pd.merge(df, df_cluster_assignment, left_index=True, right_index=True)
                df_merged = df_merged = df_merged[[unique_id] + cluster_features + ["Cluster"]]
                st.write("Cluster Centriods")
                st.write(df_centroids)
                st.write(df_merged.head(10))
                st.write("Cluster Visualization")
                plt.figure(figsize=(10, 2))  # Adjust height for a better 1D visualization
                plt.scatter(X[cluster_features], [0] * len(X), c=kmeans.labels_, cmap='viridis', alpha=0.6, edgecolors='w', s=100)

                # Mark centroids with 'X'
                plt.scatter(kmeans.cluster_centers_[:, 0], [0] * k, s=300, c='red', marker='X', edgecolors='k')

                plt.title('K-Means Clustering Visualization')
                plt.xlabel('StudyTimeWeekly')
                plt.yticks([])  # Remove y-axis labels since it's not relevant
                plt.grid(True)
                st.pyplot(plt)
            elif len(cluster_features) == 2:
                X = df[cluster_features]
                k = st.number_input("Select the number of clusters k", min_value=1, step=1)
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X)
                centroids = kmeans.cluster_centers_
                df_centroids = pd.DataFrame(centroids, columns=cluster_features)
                df_centroids['Cluster'] = range(1, len(centroids) + 1)  # Adding cluster numbers

                # Create a DataFrame for cluster assignments
                df_cluster_assignment = pd.DataFrame({
                "Cluster": kmeans.labels_,               # Cluster labels for each point
                unique_id: df[unique_id],                # Include the unique identifier
                })

                # Merge the original features with the cluster assignment
                df_cluster_assignment = pd.concat([df_cluster_assignment, X], axis=1)
                st.write("Cluster Centriods")
                st.write(df_centroids)
                st.write(df_cluster_assignment.head(10))
                st.write("Cluster Visualization")
                plt.figure(figsize=(10, 6))
                plt.scatter(X[cluster_features[0]], X[cluster_features[1]], c=kmeans.labels_, cmap='viridis', alpha=0.6, edgecolors='w', s=100)

                # Plot the centroids
                plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', edgecolors='k')

                # Adding labels and title
                plt.title('K-Means Clustering Visualization')
                plt.xlabel(cluster_features[0])
                plt.ylabel(cluster_features[1])
                plt.grid(True)
                st.pyplot(plt)

            elif len(cluster_features) > 2:
                X = df[cluster_features]
                k = st.number_input("Select the number of clusters k", min_value=1, step=1)
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X)
                centroids = kmeans.cluster_centers_
                df_centroids = pd.DataFrame(centroids, columns=cluster_features)
                df_centroids['Cluster'] = range(1, len(centroids) + 1)  # Adding cluster numbers

                # Create a DataFrame for cluster assignments
                df_cluster_assignment = pd.DataFrame({
                "Cluster": kmeans.labels_,               # Cluster labels for each point
                unique_id: df[unique_id],                # Include the unique identifier
                })

                # Merge the original features with the cluster assignment
                df_cluster_assignment = pd.concat([df_cluster_assignment, X], axis=1)
                st.write("Cluster Centriods")
                st.write(df_centroids)
                st.write(df_cluster_assignment.head(10))

    elif analysis_mode == "Predictive Analysis":
        st.subheader("Predictive Analysis")
        df_columns = df.columns.tolist()
        prediction_variable = st.selectbox("Which variable would you like to predict?", df_columns)
        data_types = identify_variable_types(df)
        if prediction_variable in data_types['continuous']:
            st.write("Based on the selected variable to be predicted, it has been identified as Continoues. Based on this a regression prediction task will be done.")
            task = "Regression"
        else:
            st.write("Based on the selected variable to be prediction, it has been identified as Discrete. Based on this a classification prediction task will be done.")
            task = "Classification"
        role = """You are an experienced Data Scientist and your role is to suggest which predictive machine learning method
        is most suitable for the data and task at hand. You will take into account the variables available, the variable to be predicted and a preview of the dataset to make the decision. If the chosen variable to be predicted is not a suitable variable also make sure to outline this.
        You will provide the most suitable methods for the task in a concise manner while givin a short justification."""
        data_preview = df.head(10) 
        prompt_text = role + "'''" + f"The variable that the user wants to predict is {prediction_variable}" + f"The variables in the dataset are {df_columns}" + f"Data Preview:{data_preview}"
        output = llama31_70b(prompt_text)
        st.write("**Predictive Method Recommendation**")
        st.write(output)
        features = st.multiselect("Select the features to be used in developing the prediction model. If you are not sure of the best features to be selected select 'auto' to be automatically select the best features.", ["Auto"] + df_columns)
        method = st.selectbox("Select which machine learning methods you would like to use build the prediction model", ["SVM", "XGBoost", "Gradient Boosting", "Random Forest", "KNN", "Nueral Network"])
        if task == 'Classification':
            if method == 'SVM':
                X = df[features]
                y = df[prediction_variable]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                start_training = st.button("Start Training!")
                if start_training == True:
                    classifier = SVC()
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(1,0.5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 5}, cbar=False)
                    plt.xlabel('Predicted Labels', fontdict= {'fontsize': 4})
                    plt.ylabel('True Labels', {'fontsize': 4})
                    plt.xticks(fontsize=4)
                    plt.yticks(fontsize=4)
                    plt.title('Confusion Matrix Heatmap', fontdict= {'fontsize': 4})
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    styled_report = report_df.style.background_gradient(cmap='viridis').format(precision=2)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(plt)
                    with col2:
                        st.write(styled_report)
                    template = "You will interpret the results from the SVM model highlighting only the key findings:"
                    prompt_text = template +  "'''" + str(accuracy) + "'''" + str(report)
                    output = llama31_70b(prompt_text)
                    st.write(output)

            elif method == 'XGBoost':
                training_method = st.selectbox("Select if you want to train the XGBoost model using the default parameter or utilize hyperparamete tuning to optimize the model", ["Default Parameters", "Optimize Parameter via Hyperparameter Tuning"], help="Optimizing Parameters via Hyperparameter Tuning will result in a longer training period")
    
                if training_method == "Default Parameters":
                    start_training = st.button("Start Training!")
                    if start_training == True:
                        X = df[features]
                        y = df[prediction_variable]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = xgboost.XGBClassifier()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        cm = confusion_matrix(y_test, y_pred)
                        plt.figure(figsize=(1,0.5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 5}, cbar=False)
                        plt.xlabel('Predicted Labels', fontdict= {'fontsize': 4})
                        plt.ylabel('True Labels', {'fontsize': 4})
                        plt.xticks(fontsize=4)
                        plt.yticks(fontsize=4)
                        plt.title('Confusion Matrix Heatmap', fontdict= {'fontsize': 4})
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        styled_report = report_df.style.background_gradient(cmap='viridis').format(precision=2)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(plt)
                        with col2:
                            st.write(styled_report)
                        template = "You will interpret the results from the XGBoost model highlighting only the key findings:"
                        prompt_text = template +  "'''" + str(accuracy) + "'''" + str(report)
                        output = llama31_70b(prompt_text)
                        st.write(output)
                        
                else:
                    st.write("""
                        The following parameters can be tuned for hyperparameter optimization in the XGBoost algorithm:

                        1. **Learning Rate:** Controls the step size at each iteration while moving toward a minimum of the loss function. A smaller learning rate requires more iterations but can lead to better model performance.

                        2. **Maximum Depth:** Determines the maximum depth of a tree. Increasing depth allows the model to capture more complex patterns but may lead to overfitting if set too high.

                        3. **Number of Estimators:** Specifies the number of trees to be built in the model. A higher number of estimators can improve model accuracy but may increase computation time and risk overfitting.

                        4. **Subsample:** Represents the fraction of samples to be used for each tree. Setting a value less than 1.0 introduces randomness, which helps prevent overfitting by ensuring that not all data points are used in each iteration.
                        """)
                    approach = st.selectbox("Select if you to manually choose the parameters or automcatically select the best hyperparamters", ["Manual", "Auto"])
                    parameters = st.multiselect("Select the parameters to be tuned", ["Learning Rate", "Maximum Depth", "Number of Estimators", "Subsample"])
                    if approach == "Manual":
                        param_grid = {}
                        if "Learning Rate" in parameters:
                            param_grid['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
                        else:
                            param_grid['learning_rate'] = 0.1
                        if "Maximum Depth" in parameters:
                            param_grid['max_depth'] = st.slider("Maximum Depth", 1, 10, 3, 1)
                        else:
                            param_grid['max_depth'] = 6
                        if "Number of Estimators" in parameters:
                            param_grid['n_estimators'] = st.slider("Number of Estimators", 50, 500, 100, 50)
                        else:
                            param_grid['n_estimators'] = 100
                        if "Subsample" in parameters:
                            param_grid['subsample'] = st.slider("Subsample", 0.5, 1.0, 0.8, 0.1)
                        else:
                            param_grid['subsample'] = 0.5

                        start_training = st.button("Start Training!")
                        if start_training == True:  
                            X = df[features]
                            y = df[prediction_variable]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            model = xgboost.XGBClassifier(learning_rate=param_grid['learning_rate'], 
                                max_depth= param_grid['max_depth'], 
                                n_estimators=param_grid['n_estimators'], 
                                subsample=param_grid['subsample'])
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            cm = confusion_matrix(y_test, y_pred)
                            plt.figure(figsize=(1,0.5))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 5}, cbar=False)
                            plt.xlabel('Predicted Labels', fontdict= {'fontsize': 4})
                            plt.ylabel('True Labels', {'fontsize': 4})
                            plt.xticks(fontsize=4)
                            plt.yticks(fontsize=4)
                            plt.title('Confusion Matrix Heatmap', fontdict= {'fontsize': 4})
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            styled_report = report_df.style.background_gradient(cmap='viridis').format(precision=2)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.pyplot(plt)
                            with col2:
                                st.write(styled_report)
                            template = "You will interpret the results from the XGBoost model highlighting only the key findings:"
                            prompt_text = template +  "'''" + str(accuracy) + "'''" + str(report)
                            output = llama31_70b(prompt_text)
                            st.write(output)

                    else:
                        param_grid = {}
                        if "Learning Rate" in parameters:
                        # Use multiselect for selecting multiple learning rates
                            selected_learning_rates = st.multiselect("Select Learning Rates", [0.01, 0.02, 0.05, 0.1, 0.2])
                            if selected_learning_rates:
                                param_grid['learning_rate'] = selected_learning_rates

                        if "Maximum Depth" in parameters:
                        # Use multiselect for selecting multiple maximum depth values
                            selected_max_depths = st.multiselect("Select Maximum Depths", [2, 3, 4, 5, 6])
                            if selected_max_depths:
                                param_grid['max_depth'] = selected_max_depths

                        if "Number of Estimators" in parameters:
                            # Use multiselect for selecting multiple numbers of estimators
                            selected_estimators = st.multiselect("Select Number of Estimators", [10, 100, 1000, 2000, 4000])
                            if selected_estimators:
                                param_grid['n_estimators'] = selected_estimators

                        if "Subsample" in parameters:
                            # Use multiselect for selecting multiple subsample values
                            selected_subsamples = st.multiselect("Select Subsample Values", [0.01, 0.1, 0.5, 0.8])
                            if selected_subsamples:
                                param_grid['subsample'] = selected_subsamples
                        start_training = st.button("Start Training!")
                        if start_training == True:
                            X = df[features]
                            y = df[prediction_variable]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            model = GridSearchCV(estimator=xgboost.XGBClassifier( subsample=0.5, 
                                    colsample_bytree=0.5, 
                                    eval_metric='auc',
                                    use_label_encoder = False), 
                                    param_grid=param_grid, 
                                    cv=5, 
                                    scoring= ['accuracy', 'roc_auc'],
                                    refit='accuracy')
                            model.fit(X_train, y_train)
                            
                            optimal_params = model.best_params_

                            if "Subsample" in parameters:
                                optimal_learning_subsample = optimal_params.get("subsample", "Not Selected")
                                st.write("**Optimal Subsample:**", optimal_learning_subsample)

                            if "Learning Rate" in parameters:
                                optimal_learning_learningrate = optimal_params.get("learning_rate", "Not Selected")
                                st.write("**Optimal Learning Rate:**", optimal_learning_learningrate)

                            if "Number of Estimators" in parameters:
                                optimal_learning_estimators = optimal_params.get("n_estimators", "Not Selected")
                                st.write("**Optimal Number of Estimators:**", optimal_learning_estimators)

                            if "Maximum Depth" in parameters:
                                optimal_learning_maxdepth = optimal_params.get("max_depth", "Not Selected")
                                st.write("**Optimal Maximum Depth:**", optimal_learning_maxdepth)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            cm = confusion_matrix(y_test, y_pred)
                            plt.figure(figsize=(1,0.5))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 5}, cbar=False)
                            plt.xlabel('Predicted Labels', fontdict= {'fontsize': 4})
                            plt.ylabel('True Labels', {'fontsize': 4})
                            plt.xticks(fontsize=4)
                            plt.yticks(fontsize=4)
                            plt.title('Confusion Matrix Heatmap', fontdict= {'fontsize': 4})
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            styled_report = report_df.style.background_gradient(cmap='viridis').format(precision=2)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.pyplot(plt)
                            with col2:
                                st.write(styled_report)
                            template = "You will interpret the results from the XGBoost model highlighting only the key findings:"
                            prompt_text = template +  "'''" + str(accuracy) + "'''" + str(report)
                            output = llama31_70b(prompt_text)
                            st.write(output)

            elif method == 'Random Forest':
                training_method = st.selectbox("Select if you want to train the XGBoost model using the default parameter or utilize hyperparamete tuning to optimize the model", ["Default Parameters", "Optimize Parameter via Hyperparameter Tuning"], help="Optimizing Parameters via Hyperparameter Tuning will result in a longer training period")
                if training_method == "Default Parameters":
                    X = df[features]
                    y = df[prediction_variable]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    start_training = st.button("Start Training!")
                    if start_training == True:
                        model = RandomForestClassifier()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        cm = confusion_matrix(y_test, y_pred)
                        plt.figure(figsize=(1,0.5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 5}, cbar=False)
                        plt.xlabel('Predicted Labels', fontdict= {'fontsize': 4})
                        plt.ylabel('True Labels', {'fontsize': 4})
                        plt.xticks(fontsize=4)
                        plt.yticks(fontsize=4)
                        plt.title('Confusion Matrix Heatmap', fontdict= {'fontsize': 4})
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        styled_report = report_df.style.background_gradient(cmap='viridis').format(precision=2)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(plt)
                        with col2:
                            st.write(styled_report)
                        template = "You will interpret the results from the Random Forest model highlighting only the key findings:"
                        prompt_text = template +  "'''" + str(accuracy) + "'''" + str(report)
                        output = llama31_70b(prompt_text)
                        st.write(output)

                else:
                    st.write("""
                        The following parameters can be tuned for hyperparameter optimization in the Random Forest algorithm:

                        1. **Maximum Depth:** Determines the maximum depth of each tree in the forest. A deeper tree can capture more complex patterns in the data but may lead to overfitting if the depth is too high. Setting a limit on the depth of the trees helps control the model complexity and prevents overfitting.

                        2. **Number of Estimators:** Specifies the number of trees to be built in the model (i.e., the size of the forest). A higher number of estimators can lead to better model performance by reducing variance and improving accuracy. However, this also increases computation time, so a balance must be struck between accuracy and computational efficiency.

                        3. **Minimum Samples Split:** The minimum number of samples required to split an internal node. This parameter controls how deep the tree grows by ensuring that a node is split only if it contains at least the specified number of samples. Higher values prevent the model from learning overly specific patterns, thus reducing the risk of overfitting.

                        4. **Minimum Samples Leaf:** The minimum number of samples required to be at a leaf node. This parameter helps prevent the model from creating nodes that contain very few data points, which can lead to overfitting. Setting this value higher forces the algorithm to generalize more by considering a larger number of samples at each leaf.

                        5. **Maximum Features:** The maximum number of features considered for splitting a node. By limiting the number of features, this parameter introduces randomness into the model, making it more robust to overfitting and increasing the model's ability to generalize. The `sqrt` of the total number of features is often used as a default value for classification tasks, while `log2` is another common choice.

                        """)
                    approach = st.selectbox("Select if you to manually choose the parameters or automcatically select the best hyperparamters", ["Manual", "Auto"])
                    parameters = st.multiselect("Select the parameters to be tuned", ["Maximum Depth", "Number of Estimators", "Minimum Sample Split", "Minimum Samples Leaf", "Maximum Features"])
                    if approach == "Manual":
                        param_grid = {}
                        if "Maximum Depth" in parameters:
                            param_grid['max_depth'] = st.slider("Maximum Depth", 1, 20, 10)  # Changed to correct order (min, max, default)
                        else:
                            param_grid['max_depth'] = None

                        if "Maximum Features" in parameters:
                            param_grid['max_features'] = st.selectbox("Maximum Features", ['sqrt', 'log2', 'None'])
                        else:
                            param_grid['max_features'] = 'sqrt'  # Corrected key

                        if "Number of Estimators" in parameters:
                            param_grid['n_estimators'] = st.slider("Number of Estimators", 50, 1000, 100)  # Changed to correct order (min, max, default)
                        else:
                            param_grid['n_estimators'] = 100

                        if "Minimum Samples Leaf" in parameters:
                            param_grid['min_samples_leaf'] = st.slider("Minimum Samples Leaf", 1, 5, 1)  # Using integers instead
                        else:
                            param_grid['min_samples_leaf'] = 1

                        if "Minimum Sample Split" in parameters:
                            param_grid['min_samples_split'] = st.slider("Minimum Sample Split", 2, 5, 2)  # Corrected the typo and range
                        else:
                            param_grid['min_samples_split'] = 2

                        X = df[features]
                        y = df[prediction_variable]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        
                        start_training = st.button("Start Training!")
                        if start_training == True:
                            model = RandomForestClassifier(max_depth= param_grid['max_depth'], 
                                                    max_features= param_grid['max_features'],  
                                                    min_samples_leaf= param_grid['min_samples_leaf'], 
                                                    min_samples_split=param_grid['min_samples_split'], 
                                                    n_estimators=param_grid['n_estimators'])
                            model.fit(X_train, y_train)
                    else:
                        X = df[features]
                        y = df[prediction_variable]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        param_grid = {}
                        if "Maximum Depth" in parameters:
                            # Use multiselect for selecting multiple maximum depth values
                            selected_max_depths = st.multiselect("Select Maximum Depths", [1, 2, 3, 4, 5, 10, 15, 20])
                            if selected_max_depths:
                                param_grid['max_depth'] = selected_max_depths
                        else:
                            param_grid['max_depth'] = [None]  # Default to None if not specified

                        if "Maximum Features" in parameters:
                            # Use multiselect for selecting multiple maximum features values
                            selected_max_features = st.multiselect("Select Maximum Features", ['sqrt', 'log2', None])
                            if selected_max_features:
                                param_grid['max_features'] = selected_max_features
                        else:
                            param_grid['max_features'] = ['sqrt']  # Default to 'sqrt' if not specified

                        if "Number of Estimators" in parameters:
                            # Use multiselect for selecting multiple numbers of estimators
                            selected_estimators = st.multiselect("Select Number of Estimators", [50, 100, 200, 500, 1000])
                            if selected_estimators:
                                param_grid['n_estimators'] = selected_estimators
                        else:
                            param_grid['n_estimators'] = [100]  # Default to 100 if not specified

                        if "Minimum Samples Leaf" in parameters:
                            # Use multiselect for selecting multiple minimum samples leaf values
                            selected_min_samples_leaf = st.multiselect("Select Minimum Samples Leaf", [1, 2, 3, 4, 5])
                            if selected_min_samples_leaf:
                                param_grid['min_samples_leaf'] = selected_min_samples_leaf
                        else:
                            param_grid['min_samples_leaf'] = [1]  # Default to 1 if not specified

                        if "Minimum Sample Split" in parameters:
                            # Use multiselect for selecting multiple minimum sample split values
                            selected_min_samples_split = st.multiselect("Select Minimum Sample Split", [2, 3, 4, 5])
                            if selected_min_samples_split:
                                param_grid['min_samples_split'] = selected_min_samples_split
                        else:
                            param_grid['min_samples_split'] = [2]  # Default to 2 if not specified
                        
                        # Assuming model.best_params_ is already defined and contains the optimal parameters
                        start_training = st.button("Start Training!")
                        if start_training == True:
                            model = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, 
                                            scoring= ['accuracy', 'roc_auc'],
                                            refit='accuracy')
                            model.fit(X_train, y_train)
                            optimal_params = model.best_params_

                            if "Maximum Depth" in parameters:
                                optimal_max_depth = optimal_params.get("max_depth", "Not Selected")
                                st.write("**Optimal Maximum Depth:**", optimal_max_depth)

                            if "Maximum Features" in parameters:
                                optimal_max_features = optimal_params.get("max_features", "Not Selected")
                                st.write("**Optimal Maximum Features:**", optimal_max_features)

                            if "Number of Estimators" in parameters:
                                optimal_estimators = optimal_params.get("n_estimators", "Not Selected")
                                st.write("**Optimal Number of Estimators:**", optimal_estimators)

                            if "Minimum Samples Leaf" in parameters:
                                optimal_min_samples_leaf = optimal_params.get("min_samples_leaf", "Not Selected")
                                st.write("**Optimal Minimum Samples Leaf:**", optimal_min_samples_leaf)

                            if "Minimum Sample Split" in parameters:
                                optimal_min_samples_split = optimal_params.get("min_samples_split", "Not Selected")
                                st.write("**Optimal Minimum Sample Split:**", optimal_min_samples_split)
                                
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            cm = confusion_matrix(y_test, y_pred)
                            plt.figure(figsize=(1,0.5))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 5}, cbar=False)
                            plt.xlabel('Predicted Labels', fontdict= {'fontsize': 4})
                            plt.ylabel('True Labels', {'fontsize': 4})
                            plt.xticks(fontsize=4)
                            plt.yticks(fontsize=4)
                            plt.title('Confusion Matrix Heatmap', fontdict= {'fontsize': 4})
                            report = classification_report(y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            styled_report = report_df.style.background_gradient(cmap='viridis').format(precision=2)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.pyplot(plt)
                            with col2:
                                st.write(styled_report)
                            template = "You will interpret the results from the Random Forest model highlighting only the key findings:"
                            prompt_text = template +  "'''" + str(accuracy) + "'''" + str(report)
                            output = llama31_70b(prompt_text)
                            st.write(output)
            else:
                st.write(f"The {method} method has not bee released yet. Stay Tuned!")
        else:
            if method == 'SVM':
                X = df[features]
                y = df[prediction_variable]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Use SVR for regression
                
                start_training = st.button("Start Training!")
                if start_training == True:
                    regressor = SVR()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)

                # Evaluate regression performance
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    # Plotting predicted vs actual values
                    plt.figure(figsize=(5, 5))
                    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
                    plt.xlabel('Actual Values')
                    plt.ylabel('Predicted Values')
                    plt.title('Actual vs Predicted Values')
                    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(plt)
                    with col2:
                        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                        st.write(f"R-squared (R2): {r2:.2f}")
                    
                    # Generate text output using a model like Llama31_70b if applicable
                    template = "You will interpret the results from the SVM regression model highlighting only the key findings:"
                    prompt_text = template + f"\nMSE: {mse}\nMAE: {mae}\nR2: {r2}"
                    output = llama31_70b(prompt_text)
                    st.write(output)  
            elif method == 'XGBoost':
                training_method = st.selectbox(
                "Select if you want to train the XGBoost model using the default parameter or utilize hyperparameter tuning to optimize the model",
                ["Default Parameters", "Optimize Parameter via Hyperparameter Tuning"],
                help="Optimizing Parameters via Hyperparameter Tuning will result in a longer training period"
            )

                if training_method == "Default Parameters":
                    X = df[features]
                    y = df[prediction_variable]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    start_training = st.button("Start Training!")
                    if start_training == True:
                        model = xgboost.XGBRegressor()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        # Visualizing predicted vs actual values
                        plt.figure(figsize=(5, 5))
                        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
                        plt.xlabel('Actual Values')
                        plt.ylabel('Predicted Values')
                        plt.title('Actual vs Predicted Values')
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(plt)
                        with col2:
                            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                            st.write(f"R-squared (R2): {r2:.2f}")

                        template = "You will interpret the results from the XGBoost regression model highlighting only the key findings:"
                        prompt_text = template + f"\nMSE: {mse}\nMAE: {mae}\nR2: {r2}"
                        output = llama31_70b(prompt_text)
                        st.write(output) 
                    
                else:
                    st.write("""
                        The following parameters can be tuned for hyperparameter optimization in the XGBoost algorithm:

                        1. **Learning Rate:** Controls the step size at each iteration while moving toward a minimum of the loss function. A smaller learning rate requires more iterations but can lead to better model performance.

                        2. **Maximum Depth:** Determines the maximum depth of a tree. Increasing depth allows the model to capture more complex patterns but may lead to overfitting if set too high.

                        3. **Number of Estimators:** Specifies the number of trees to be built in the model. A higher number of estimators can improve model accuracy but may increase computation time and risk overfitting.

                        4. **Subsample:** Represents the fraction of samples to be used for each tree. Setting a value less than 1.0 introduces randomness, which helps prevent overfitting by ensuring that not all data points are used in each iteration.
                    """)
                    
                    approach = st.selectbox(
                        "Select if you want to manually choose the parameters or automatically select the best hyperparameters",
                        ["Manual", "Auto"]
                    )
                    
                    parameters = st.multiselect(
                        "Select the parameters to be tuned",
                        ["Learning Rate", "Maximum Depth", "Number of Estimators", "Subsample"]
                    )
                    
                    if approach == "Manual":
                        param_grid = {}
                        if "Learning Rate" in parameters:
                            param_grid['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
                        else:
                            param_grid['learning_rate'] = 0.1
                        if "Maximum Depth" in parameters:
                            param_grid['max_depth'] = st.slider("Maximum Depth", 1, 10, 3, 1)
                        else:
                            param_grid['max_depth'] = 6
                        if "Number of Estimators" in parameters:
                            param_grid['n_estimators'] = st.slider("Number of Estimators", 50, 500, 100, 50)
                        else:
                            param_grid['n_estimators'] = 100
                        if "Subsample" in parameters:
                            param_grid['subsample'] = st.slider("Subsample", 0.5, 1.0, 0.8, 0.1)
                        else:
                            param_grid['subsample'] = 0.5

                        start_training = st.button("Start Training!")
                        if start_training == True:

                            X = df[features]
                            y = df[prediction_variable]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            model = xgboost.XGBRegressor(
                                learning_rate=param_grid['learning_rate'],
                                max_depth=param_grid['max_depth'],
                                n_estimators=param_grid['n_estimators'],
                                subsample=param_grid['subsample']
                            )
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            mse = mean_squared_error(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)

                            # Visualizing predicted vs actual values
                            plt.figure(figsize=(5, 5))
                            sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
                            plt.xlabel('Actual Values')
                            plt.ylabel('Predicted Values')
                            plt.title('Actual vs Predicted Values')
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.pyplot(plt)
                            with col2:
                                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                                st.write(f"R-squared (R2): {r2:.2f}")

                            template = "You will interpret the results from the XGBoost regression model highlighting only the key findings:"
                            prompt_text = template + f"\nMSE: {mse}\nMAE: {mae}\nR2: {r2}"
                            output = llama31_70b(prompt_text)
                            st.write(output) 
                    else:
                        param_grid = {}
                        if "Learning Rate" in parameters:
                            selected_learning_rates = st.multiselect("Select Learning Rates", [0.01, 0.02, 0.05, 0.1, 0.2])
                            if selected_learning_rates:
                                param_grid['learning_rate'] = selected_learning_rates

                        if "Maximum Depth" in parameters:
                            selected_max_depths = st.multiselect("Select Maximum Depths", [2, 3, 4, 5, 6])
                            if selected_max_depths:
                                param_grid['max_depth'] = selected_max_depths

                        if "Number of Estimators" in parameters:
                            selected_estimators = st.multiselect("Select Number of Estimators", [10, 100, 1000, 2000, 4000])
                            if selected_estimators:
                                param_grid['n_estimators'] = selected_estimators

                        if "Subsample" in parameters:
                            selected_subsamples = st.multiselect("Select Subsample Values", [0.5, 0.6, 0.7, 0.8, 0.9])
                            if selected_subsamples:
                                param_grid['subsample'] = selected_subsamples

                        start_training = st.button("Start Training!")
                        if start_training == True:
                            X = df[features]
                            y = df[prediction_variable]
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            model = GridSearchCV(
                            estimator=xgboost.XGBRegressor(),
                            param_grid=param_grid,
                            cv=5,
                            scoring='neg_mean_squared_error',
                            refit=True
                        )
                            model.fit(X_train, y_train)
                            optimal_params = model.best_params_

                            if "Subsample" in parameters:
                                optimal_learning_subsample = optimal_params.get("subsample", "Not Selected")
                                st.write("**Optimal Subsample:**", optimal_learning_subsample)

                            if "Learning Rate" in parameters:
                                optimal_learning_learningrate = optimal_params.get("learning_rate", "Not Selected")
                                st.write("**Optimal Learning Rate:**", optimal_learning_learningrate)

                            if "Number of Estimators" in parameters:
                                optimal_learning_estimators = optimal_params.get("n_estimators", "Not Selected")
                                st.write("**Optimal Number of Estimators:**", optimal_learning_estimators)

                            if "Maximum Depth" in parameters:
                                optimal_learning_maxdepth = optimal_params.get("max_depth", "Not Selected")
                                st.write("**Optimal Maximum Depth:**", optimal_learning_maxdepth)

                # Predict and evaluate regression results
                
                            y_pred = model.predict(X_test)

                            mse = mean_squared_error(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)

                            # Visualizing predicted vs actual values
                            plt.figure(figsize=(5, 5))
                            sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
                            plt.xlabel('Actual Values')
                            plt.ylabel('Predicted Values')
                            plt.title('Actual vs Predicted Values')
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.pyplot(plt)
                            with col2:
                                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                                st.write(f"R-squared (R2): {r2:.2f}")

                            template = "You will interpret the results from the XGBoost regression model highlighting only the key findings:"
                            prompt_text = template + f"\nMSE: {mse}\nMAE: {mae}\nR2: {r2}"
                            output = llama31_70b(prompt_text)
                            st.write(output) 
            elif method == "Random Forest":
                training_method = st.selectbox(
                    "Select if you want to train the Random Forest model using the default parameter or utilize hyperparameter tuning to optimize the model",
                    ["Default Parameters", "Optimize Parameter via Hyperparameter Tuning"],
                    help="Optimizing Parameters via Hyperparameter Tuning will result in a longer training period"
                )
                
                X = df[features]
                y = df[prediction_variable]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if training_method == "Default Parameters":
                    model = RandomForestRegressor()
                    start_training = st.button("Start Training!")
                    if start_training == True:
                        model = RandomForestRegressor()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        # Visualization of actual vs predicted values
                        plt.figure(figsize=(5, 5))
                        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
                        plt.xlabel('Actual Values')
                        plt.ylabel('Predicted Values')
                        plt.title('Actual vs Predicted Values')
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(plt)
                        with col2:
                            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                            st.write(f"R-squared (R2): {r2:.2f}")

                        template = "You will interpret the results from the Random Forest regression model highlighting only the key findings:"
                        prompt_text = template + f"\nMSE: {mse}\nMAE: {mae}\nR2: {r2}"
                        output = llama31_70b(prompt_text)
                        st.write(output)
                
                else:
                    st.write("""
                        The following parameters can be tuned for hyperparameter optimization in the Random Forest algorithm:

                        1. **Maximum Depth:** Determines the maximum depth of each tree in the forest. A deeper tree can capture more complex patterns in the data but may lead to overfitting if the depth is too high. Setting a limit on the depth of the trees helps control the model complexity and prevents overfitting.

                        2. **Number of Estimators:** Specifies the number of trees to be built in the model (i.e., the size of the forest). A higher number of estimators can lead to better model performance by reducing variance and improving accuracy. However, this also increases computation time, so a balance must be struck between accuracy and computational efficiency.

                        3. **Minimum Samples Split:** The minimum number of samples required to split an internal node. This parameter controls how deep the tree grows by ensuring that a node is split only if it contains at least the specified number of samples. Higher values prevent the model from learning overly specific patterns, thus reducing the risk of overfitting.

                        4. **Minimum Samples Leaf:** The minimum number of samples required to be at a leaf node. This parameter helps prevent the model from creating nodes that contain very few data points, which can lead to overfitting. Setting this value higher forces the algorithm to generalize more by considering a larger number of samples at each leaf.

                        5. **Maximum Features:** The maximum number of features considered for splitting a node. By limiting the number of features, this parameter introduces randomness into the model, making it more robust to overfitting and increasing the model's ability to generalize. The `sqrt` of the total number of features is often used as a default value for classification tasks, while `log2` is another common choice.
                    """)
                    
                    approach = st.selectbox(
                        "Select if you want to manually choose the parameters or automatically select the best hyperparameters",
                        ["Manual", "Auto"]
                    )
                    
                    parameters = st.multiselect(
                        "Select the parameters to be tuned",
                        ["Maximum Depth", "Number of Estimators", "Minimum Sample Split", "Minimum Samples Leaf", "Maximum Features"]
                    )
                    
                    if approach == "Manual":
                        param_grid = {}
                        if "Maximum Depth" in parameters:
                            param_grid['max_depth'] = st.slider("Maximum Depth", 1, 20, 10)
                        else:
                            param_grid['max_depth'] = None

                        if "Maximum Features" in parameters:
                            param_grid['max_features'] = st.selectbox("Maximum Features", ['sqrt', 'log2', 'None'])
                        else:
                            param_grid['max_features'] = 'sqrt'

                        if "Number of Estimators" in parameters:
                            param_grid['n_estimators'] = st.slider("Number of Estimators", 50, 1000, 100)
                        else:
                            param_grid['n_estimators'] = 100

                        if "Minimum Samples Leaf" in parameters:
                            param_grid['min_samples_leaf'] = st.slider("Minimum Samples Leaf", 1, 5, 1)
                        else:
                            param_grid['min_samples_leaf'] = 1

                        if "Minimum Samples Split" in parameters:
                            param_grid['min_samples_split'] = st.slider("Minimum Sample Split", 2, 5, 2)
                        else:
                            param_grid['min_samples_split'] = 2

                        
                        start_training = st.button("Start Training!")
                        if start_training == True:
                            model = RandomForestRegressor(
                            max_depth=param_grid['max_depth'],
                            max_features=param_grid['max_features'],
                            min_samples_leaf=param_grid['min_samples_leaf'],
                            min_samples_split=param_grid['min_samples_split'],
                            n_estimators=param_grid['n_estimators']
                        )
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            mse = mean_squared_error(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)

                            # Visualization of actual vs predicted values
                            plt.figure(figsize=(5, 5))
                            sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
                            plt.xlabel('Actual Values')
                            plt.ylabel('Predicted Values')
                            plt.title('Actual vs Predicted Values')
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.pyplot(plt)
                            with col2:
                                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                                st.write(f"R-squared (R2): {r2:.2f}")

                            template = "You will interpret the results from the Random Forest regression model highlighting only the key findings:"
                            prompt_text = template + f"\nMSE: {mse}\nMAE: {mae}\nR2: {r2}"
                            output = llama31_70b(prompt_text)
                            st.write(output)




                    
                    else:
                        param_grid = {}
                        if "Maximum Depth" in parameters:
                            selected_max_depths = st.multiselect("Select Maximum Depths", [1, 2, 3, 4, 5, 10, 15, 20])
                            if selected_max_depths:
                                param_grid['max_depth'] = selected_max_depths
                        else:
                            param_grid['max_depth'] = [None]

                        if "Maximum Features" in parameters:
                            selected_max_features = st.multiselect("Select Maximum Features", ['sqrt', 'log2', None])
                            if selected_max_features:
                                param_grid['max_features'] = selected_max_features
                        else:
                            param_grid['max_features'] = ['sqrt']

                        if "Number of Estimators" in parameters:
                            selected_estimators = st.multiselect("Select Number of Estimators", [50, 100, 200, 500, 1000])
                            if selected_estimators:
                                param_grid['n_estimators'] = selected_estimators
                        else:
                            param_grid['n_estimators'] = [100]

                        if "Minimum Samples Leaf" in parameters:
                            selected_min_samples_leaf = st.multiselect("Select Minimum Samples Leaf", [1, 2, 3, 4, 5])
                            if selected_min_samples_leaf:
                                param_grid['min_samples_leaf'] = selected_min_samples_leaf
                        else:
                            param_grid['min_samples_leaf'] = [1]

                        if "Minimum Samples Split" in parameters:
                            selected_min_samples_split = st.multiselect("Select Minimum Sample Split", [2, 3, 4, 5])
                            if selected_min_samples_split:
                                param_grid['min_samples_split'] = selected_min_samples_split
                        else:
                            param_grid['min_samples_split'] = [2]

                        
                        start_training = st.button("Start Training!")
                        if start_training == True:
                            model = GridSearchCV(
                            estimator=RandomForestRegressor(),
                            param_grid=param_grid,
                            cv=5,
                            scoring='neg_mean_squared_error',
                            refit=True
                        )
                            model.fit(X_train, y_train)
                            optimal_params = model.best_params_

                            if "Maximum Depth" in parameters:
                                optimal_max_depth = optimal_params.get("max_depth", "Not Selected")
                                st.write("**Optimal Maximum Depth:**", optimal_max_depth)

                            if "Maximum Features" in parameters:
                                optimal_max_features = optimal_params.get("max_features", "Not Selected")
                                st.write("**Optimal Maximum Features:**", optimal_max_features)

                            if "Number of Estimators" in parameters:
                                optimal_estimators = optimal_params.get("n_estimators", "Not Selected")
                                st.write("**Optimal Number of Estimators:**", optimal_estimators)

                            if "Minimum Samples Leaf" in parameters:
                                optimal_min_samples_leaf = optimal_params.get("min_samples_leaf", "Not Selected")
                                st.write("**Optimal Minimum Samples Leaf:**", optimal_min_samples_leaf)

                            if "Minimum Samples Split" in parameters:
                                optimal_min_samples_split = optimal_params.get("min_samples_split", "Not Selected")
                                st.write("**Optimal Minimum Sample Split:**", optimal_min_samples_split)

                        # Predictions and evaluation
                            y_pred = model.predict(X_test)

                            mse = mean_squared_error(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)

                            # Visualization of actual vs predicted values
                            plt.figure(figsize=(5, 5))
                            sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
                            plt.xlabel('Actual Values')
                            plt.ylabel('Predicted Values')
                            plt.title('Actual vs Predicted Values')
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.pyplot(plt)
                            with col2:
                                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                                st.write(f"R-squared (R2): {r2:.2f}")

                            template = "You will interpret the results from the Random Forest regression model highlighting only the key findings:"
                            prompt_text = template + f"\nMSE: {mse}\nMAE: {mae}\nR2: {r2}"
                            output = llama31_70b(prompt_text)
                            st.write(output)
            else:
                st.write(f"The {method} method has not bee released yet. Stay Tuned!")


    elif analysis_mode == "Data Vizualisation":
        st.subheader("Data Vizualisation")
        st.write(f"The {analysis_mode} analysis mode has not bee released yet. Stay Tuned!")
        


    elif analysis_mode == "Text Analysis":
        st.subheader("Text Analysis")
        st.write(f"The {analysis_mode} analysis mode has not bee released yet. Stay Tuned!")


    elif analysis_mode == "Dimensionality Reduction":
        st.subheader("Dimensionality Reduction")
        st.write(f"The {analysis_mode} analysis mode has not bee released yet. Stay Tuned!")

    elif analysis_mode == "Hypothesis Testing":
        st.subheader("Hypothesis Testing")
        st.write(f"The {analysis_mode} analysis mode has not bee released yet. Stay Tuned!")








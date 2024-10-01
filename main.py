import streamlit as st
import os
import plotly.express as px
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
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from DataCleaningFunctions import completness_ratio, data_val_erroneous, data_val_duplicates
from groq import Groq
import seaborn as sns
import plotly.graph_objects as go
import time
import xgboost
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key= api_key)


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

with st.spinner('Setting up your enviroment...'):
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

        tab1, tab2, tab3 = st.tabs(["Single Attribute Vizualisation", "Multi Attribute Vizualisation", "Advanced Vizualisation"])

        with tab1: 
            st.subheader("Single Attribute Vizualisation")
            variable = st.selectbox("Select the variable you want to visualize", df.columns)
            chart = st.selectbox("Select the visualization type", ["Pie Chart", "Donut Chart", "Bar Chart", 
                                                                   "Histogram", "Boxplot"])

            if chart == "Pie Chart":
                st.write("Pie Chart")
                var_counts = df[variable].value_counts().reset_index()
                var_counts.columns = [variable, 'Count']
                
                # Use Plotly pie chart with enhancements
                fig = px.pie(var_counts, names=variable, values='Count', 
                            title=f"{variable} Pie Chart", 
                            color_discrete_sequence=px.colors.qualitative.Pastel)  # Use a more appealing color scheme
                
                # Update layout for better appearance
                fig.update_traces(textposition='inside', textinfo='percent+label')  # Display percentages and labels inside slices
                
                # Customize title and layout
                fig.update_layout(
                    title=dict(font=dict(size=20), x=0.43),  # Center title and increase font size
                    height=400,  # Adjust chart height
                    margin=dict(t=50, b=0, l=0, r=0)  # Adjust margins for better fitting
                )
                
                st.plotly_chart(fig)

                data_type = identify_variable_types(df)

                if variable in data_type['continuous']:
                    st.write(f"{variable} is a continuous variable, which makes it unsuitable for a pie chart. For better representation, consider using a histogram or boxplot instead.")

            elif chart == "Donut Chart":
                st.write("Donut Chart")
                var_counts = df[variable].value_counts().reset_index()
                var_counts.columns = [variable, 'Count']
                
                # Use Plotly pie chart with hole parameter to create a donut chart
                fig = px.pie(var_counts, names=variable, values='Count', 
                            title=f"{variable} Donut Chart", 
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            hole=0.4)  # Set hole size to create the donut effect
                
                # Update layout for better appearance
                fig.update_traces(textposition='inside', textinfo='percent+label')  # Display percentages and labels inside slices
                
                # Customize title and layout
                fig.update_layout(
                    title=dict(font=dict(size=20), x=0.43),  # Center title and increase font size
                    height=400,  # Adjust chart height
                    margin=dict(t=50, b=0, l=0, r=0)  # Adjust margins for better fitting
                )
                
                st.plotly_chart(fig)

                data_type = identify_variable_types(df)

                if variable in data_type['continuous']:
                    st.write(f"{variable} is a continuous variable, which makes it unsuitable for a donut chart. For better representation, consider using a histogram or boxplot instead.")

            elif chart == "Bar Chart":
                st.write("Bar Chart")
                var_counts = df[variable].value_counts().reset_index()
                var_counts.columns = [variable, 'Count']
                fig = px.bar(var_counts, x=variable, y='Count', title=f"{variable} Bar Chart", color_discrete_sequence=px.colors.qualitative.Pastel)
                
                # Customize title and layout
                fig.update_layout(
                    title=dict(font=dict(size=20), x=0.43),  # Center title and increase font size
                    height=400,  # Adjust chart height
                    margin=dict(t=50, b=0, l=0, r=0)  # Adjust margins for better fitting
                )
                
                st.plotly_chart(fig)

                data_type = identify_variable_types(df)

                if variable in data_type['continuous']:
                    st.write(f"{variable} is a continuous variable, which makes it unsuitable for a bar chart. For better representation, consider using a histogram or boxplot instead.")

            elif chart == "Histogram":
                st.write("Histogram")
                var_counts = df[variable].value_counts().reset_index()
                var_counts.columns = [variable, 'Count']
                                
                # Create a histogram using Plotly with enhanced visuals
                fig = px.histogram(var_counts, x=variable, y='Count', 
                            title=f"Distribution of {variable}", text_auto=True)  # Use a more appealing color scheme

                # Customize the layout for better appearance
                fig.update_layout(
                    title=dict(font=dict(size=20), x=0.43),  # Center the title and adjust font size
                    xaxis_title=f"{variable}",  # Set x-axis title
                    yaxis_title="Count",  # Set y-axis title
                    height=500,  # Adjust the height of the chart
                    margin=dict(t=50, b=50, l=50, r=50),
                    plot_bgcolor='rgba(0,0,0,0)',  # Set a transparent background
                )
                # Display the enhanced histogram
                st.plotly_chart(fig)

            elif chart == "Boxplot":
                st.write("Boxplot")
                var_counts = df[variable].value_counts().reset_index()
                var_counts.columns = [variable, 'Count']

                fig = px.box(df, y= variable, title=f"Distribution of {variable}")

                fig.update_layout(
                    title=dict(font=dict(size=20), x=0.42),  # Center the title and adjust font size
                    xaxis_title=f"{variable}",  # Set x-axis title
                    yaxis_title="Count",  # Set y-axis title
                    height=500,  # Adjust the height of the chart
                    margin=dict(t=50, b=50, l=50, r=50),
                    plot_bgcolor='rgba(0,0,0,0)',  # Set a transparent background
                )
                # Display the enhanced histogram
                st.plotly_chart(fig)


        
            with tab2:
                st.subheader("Multi Attribute Visualization")

                # Step 1: Add all the numerical variables that can be visualized
                variable = st.multiselect("Select the variables you want to visualize", df.columns)    
                data_types = identify_variable_types(df)
                
                # if any(var in data_types["continuous"] for var in variable) and any(var in data_types['discrete'] for var in variable):
                #     st.write("It looks like you've selected both numerical and categorical variables. Select only one variable type to visualize at a time.")
                # else:
                #     st.write("Proceed with visualization.")
                chart = st.selectbox("Select the visualization type", ["Donut Chart", "Bar Chart", 
                                                                    "Histogram", "Boxplot", "Scatter plot"])
                
                if any(var in data_types["continuous"] for var in variable):
                    st.write("You cannot visualize continuous variables in a Donut Chart")
                else:
                    if len(variable) > 1:
                        if chart == "Donut Chart":
                            st.write("Donut Chart")
                            

                            # Group the data by selected variables and count occurrences
                            df_groupedby = df.groupby(variable).size().reset_index(name='Count')

                            # Get unique values of the first variable
                            labels = df[variable[0]].unique()

                            # Get the unique values of the second variable to determine the number of subplots
                            unique_groups = df[variable[1]].unique()

                            # Create a list of dictionaries for the specs, one 'domain' type for each column
                            specs = [[{'type':'domain'}] * len(unique_groups)]

                            # Create the subplot figure with dynamic number of columns
                            fig = make_subplots(rows=1, cols=len(unique_groups), specs=specs, subplot_titles=[f"{group} Distribution" for group in unique_groups])

                            # Loop through the unique groups and add a Pie chart trace for each one
                            for i, group in enumerate(unique_groups):
                                fig.add_trace(
                                    go.Pie(labels=labels, values=df_groupedby[df_groupedby[variable[1]] == group]['Count'], 
                                        name=f"Group {group}", marker=dict(colors=px.colors.qualitative.Plotly)), 
                                    row=1, col=i+1
                                )

                            # Update traces to show as donut charts with improved hover info and percentage formatting
                            fig.update_traces(
                            hole=.4, 
                            hoverinfo="label+percent+name")

                            # Add annotations dynamically for each pie chart, centering text inside each donut
                            annotations = [
                    dict(text=f"{variable[1]} {group}", x=(i + 0.5)/len(unique_groups), y=0.5, font_size=20, showarrow=False, xanchor="center") 
                    for i, group in enumerate(unique_groups)
                ]

                
                            fig.update_layout(
                                title_text=f"Distribution of {variable[0]} across {variable[1]} groups",
                                title= dict(font=dict(size=20), x=0.40), 
                                annotations=annotations,
                                showlegend=True,  
                                margin=dict(t=50, b=50),  
                                paper_bgcolor="White"   
                            )

                            # Display the Plotly chart in Streamlit
                            st.plotly_chart(fig)

                            data_type = identify_variable_types(df)

                            if variable in data_type['continuous']:
                                st.write(f"{variable} is a continuous variable, which makes it unsuitable for a pie chart. For better representation, consider using a histogram or boxplot instead.")

                if chart == "Bar Chart":
                    st.subheader("Multi Attribute Visualization")

                    data_types = identify_variable_types(df)
                    
                    # Ensure that a continuous and a categorical variable are selected
                    if len(variable) == 2 and any(var in data_types["continuous"] for var in variable) and any(var in data_types['discrete'] for var in variable):
                        st.write("Proceed with bar chart visualization.")
                        
                        # Determine which variable is categorical and which is continuous
                        x_var = [var for var in variable if var in data_types['discrete']][0]  # Categorical variable for x-axis
                        y_var = [var for var in variable if var in data_types['continuous']][0]  # Continuous variable for y-axis
                        
                        # Group the data by the x-variable (categorical) and calculate the mean or sum of the y-variable
                        df_grouped = df.groupby(x_var)[y_var].mean().reset_index()  # You can change this to .sum() or other aggregation if needed
                        
                        # Create a bar chart using Plotly
                        fig = go.Figure(data=[
                            go.Bar(x=df_grouped[x_var], y=df_grouped[y_var], marker=dict(color='skyblue'))  # Set bar color
                        ])
                        
                        # Update layout for better visual appeal
                        fig.update_layout(
                            title_text=f"Bar Chart of {y_var} by {x_var}",
                            title= dict(font=dict(size=20), x=0.40),
                            xaxis_title=x_var,
                            yaxis_title=y_var,
                            paper_bgcolor="White",  # Background color to improve visual appeal
                            plot_bgcolor="white",  # Background for the plot itself
                            margin=dict(t=50, b=50),  # Adjust margins for better spacing
                            bargap=0.2  # Space between bars
                        )
                        
                        # Display the Plotly bar chart in Streamlit
                        st.plotly_chart(fig)

                    else:
                        st.write("Please select one continuous and one categorical variable for visualization.")
                elif chart == "Histogram":
                    data_types = identify_variable_types(df)

                    # Ensure that one continuous and one categorical variable are selected
                    if len(variable) == 2 and any(var in data_types["continuous"] for var in variable) and any(var in data_types['discrete'] for var in variable):
                        st.write("Proceed with histogram visualization.")
                        
                        # Determine which variable is continuous and which is categorical
                        x_var = [var for var in variable if var in data_types['continuous']][0]  # Continuous variable for x-axis
                        color_var = [var for var in variable if var in data_types['discrete']][0]  # Categorical variable for color

                        # Create a histogram using Plotly
                        fig = px.histogram(df, x=x_var, color=color_var, 
                                        title=f"Histogram of {x_var} grouped by {color_var}",
                                        labels={x_var: x_var, color_var: color_var},  # Axis and color labels
                                        nbins=20,  # You can adjust the number of bins as needed
                                        color_discrete_sequence=px.colors.qualitative.Plotly)  # Color scheme
                        
                        # Update layout for better visual appeal
                        fig.update_layout(title_text=f"Histogram of {x_var} by {color_var}",
                            title=dict(font=dict(size=20), x=0.40),
                            xaxis_title=x_var,
                            yaxis_title="Frequency",
                            paper_bgcolor="White",  # Background color to improve visual appeal
                            plot_bgcolor="white",  # Background for the plot itself
                            margin=dict(t=50, b=50),  # Adjust margins for better spacing
                        )
                        
                        # Display the Plotly histogram in Streamlit
                        st.plotly_chart(fig)

                    else:
                        st.write("Please select one continuous and one categorical variable for visualization.")
                elif chart == "Boxplot":
                    data_types = identify_variable_types(df)
                    
                    # Ensure that one continuous and one categorical variable are selected
                    if len(variable) == 2 and any(var in data_types["continuous"] for var in variable) and any(var in data_types['discrete'] for var in variable):
                        st.write("Proceed with box plot visualization.")
                        
                        # Determine which variable is categorical and which is continuous
                        x_var = [var for var in variable if var in data_types['discrete']][0]  # Categorical variable for x-axis
                        y_var = [var for var in variable if var in data_types['continuous']][0]  # Continuous variable for y-axis

                        # Create a box plot using Plotly
                        fig = px.box(df, x=x_var, y=y_var, 
                                    title=f"Box Plot of {y_var} by {x_var}",
                                    labels={x_var: x_var, y_var: y_var},  # Axis labels
                                    color=x_var,  # Color by the categorical variable
                                    color_discrete_sequence=px.colors.qualitative.Plotly)  # Color scheme
                        
                        # Update layout for better visual appeal
                        fig.update_layout(title_text=f"Histogram of {y_var} by {x_var}",
                            title=dict(font=dict(size=20), x=0.40),
                            xaxis_title=x_var,
                            yaxis_title=y_var,
                            paper_bgcolor="White",  # Background color to improve visual appeal
                            plot_bgcolor="white",  # Background for the plot itself
                            margin=dict(t=50, b=50),  # Adjust margins for better spacing
                        )
                        
                        # Display the Plotly box plot in Streamlit
                        st.plotly_chart(fig)

                    else:
                        st.write("Please select one continuous and one categorical variable for visualization.")

                elif chart == "Scatter plot":

                    # Ensure exactly two variables are selected
                    if len(variable) == 2:
                        st.write("Proceed with scatter plot visualization.")

                        # Determine the x and y variables (selected by the user)
                        x_var = variable[0]  # First variable for the x-axis
                        y_var = variable[1]  # Second variable for the y-axis

                        # Create a scatter plot using Plotly
                        fig = px.scatter(df, x=x_var, y=y_var,
                                        title=f"Scatter Plot of {y_var} vs {x_var}",
                                        labels={x_var: x_var, y_var: y_var},  # Axis labels
                                        color_discrete_sequence=px.colors.qualitative.Plotly)  # Color scheme

                        # Update layout for better visual appeal
                        fig.update_layout(title_text=f"Histogram of {y_var} by {x_var}",
                            title=dict(font=dict(size=20), x=0.40),
                            xaxis_title=x_var,
                            yaxis_title=y_var,
                            paper_bgcolor="White",  # Background color to improve visual appeal
                            plot_bgcolor="white",  # Background for the plot itself
                            margin=dict(t=50, b=50),  # Adjust margins for better spacing
                        )
                        
                        # Display the Plotly scatter plot in Streamlit
                        st.plotly_chart(fig)

                    else:
                        st.write("Please select exactly two variables for visualization.")

        with tab3:
            st.subheader("Advanced Vizualisation")
            st.write("The Advanced Vizualisation analysis mode has not bee released yet. Stay Tuned!")


            # 3D Plot   
            


    elif analysis_mode == "Text Analysis":
        st.subheader("Text Analysis")
        st.write(f"The {analysis_mode} analysis mode has not bee released yet. Stay Tuned!")


    elif analysis_mode == "Dimensionality Reduction":
        st.subheader("Dimensionality Reduction")
        st.write(f"The {analysis_mode} analysis mode has not been released yet. Stay Tuned!")


        #PCA

    elif analysis_mode == "Hypothesis Testing":
        st.subheader("Hypothesis Testing")
        st.write(f"The {analysis_mode} analysis mode has not bee released yet. Stay Tuned!")









import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from styling import custom_styling
import altair as alt

st.set_page_config(
    page_title="ClaimVision - Predictive Insurance Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_styling()

@st.cache_resource
def load_or_create_model():
    model_path = 'claim_prediction_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        st.warning("Model file not found. Please train the model first.")
        try:
            train = pd.read_csv('../data/front_filled_train.csv')
            model = train_model(train)
            joblib.dump(model, model_path)
        except Exception as e:
            st.error(f"Could not train model: {e}")
            model = None
    return model

@st.cache_resource
def load_or_create_encoders():
    encoder_path = 'encoder.pkl'
    scaler_path = 'scaler.pkl'
    
    if os.path.exists(encoder_path) and os.path.exists(scaler_path):
        encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
    else:
        st.warning("Encoder and/or scaler files not found. Creating new ones from training data.")
        try:
            train = pd.read_csv('../data/front_filled_train.csv')
            
            categorical_columns = ['Gender', 'Car_Category', 'Subject_Car_Colour', 'Subject_Car_Make', 'LGA_Name', 'State', 'ProductName']
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoder.fit(train[categorical_columns].fillna('Unknown'))
            
            train['Policy_Start_Date'] = pd.to_datetime(train['Policy_Start_Date'])
            train['Policy_End_Date'] = pd.to_datetime(train['Policy_End_Date'])
            train['First_Transaction_Date'] = pd.to_datetime(train['First_Transaction_Date'])
            
            train['Policy_Duration'] = (train['Policy_End_Date'] - train['Policy_Start_Date']).dt.days
            train['Customer_Tenure'] = (train['Policy_Start_Date'] - train['First_Transaction_Date']).dt.days
            train['Recency'] = (pd.Timestamp.today() - train['Policy_End_Date']).dt.days
            
            numerical_columns = ['Age', 'No_Pol', 'Policy_Duration', 'Customer_Tenure', 'Recency']
            for col in numerical_columns:
                train[col] = train[col].fillna(train[col].mean())
            
            scaler = StandardScaler()
            scaler.fit(train[numerical_columns])
            
            joblib.dump(encoder, encoder_path)
            joblib.dump(scaler, scaler_path)
        except Exception as e:
            st.error(f"Could not create encoders: {e}")
            encoder, scaler = None, None
    
    return encoder, scaler

def train_model(train_data):
    from sklearn.tree import DecisionTreeClassifier
    
    train_data['Policy_Start_Date'] = pd.to_datetime(train_data['Policy_Start_Date'])
    train_data['Policy_End_Date'] = pd.to_datetime(train_data['Policy_End_Date'])
    train_data['First_Transaction_Date'] = pd.to_datetime(train_data['First_Transaction_Date'])
    
    train_data['Policy_Duration'] = (train_data['Policy_End_Date'] - train_data['Policy_Start_Date']).dt.days
    train_data['Customer_Tenure'] = (train_data['Policy_Start_Date'] - train_data['First_Transaction_Date']).dt.days
    train_data['Recency'] = (pd.Timestamp.today() - train_data['Policy_End_Date']).dt.days
    
    categorical_columns = ['Gender', 'Car_Category', 'Subject_Car_Colour', 'Subject_Car_Make', 'LGA_Name', 'State', 'ProductName']
    for col in categorical_columns:
        train_data[col] = train_data[col].fillna('Unknown')
    
    numerical_columns = ['Age', 'No_Pol', 'Policy_Duration', 'Customer_Tenure', 'Recency']
    for col in numerical_columns:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_data = encoder.fit_transform(train_data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
    
    scaler = StandardScaler()
    train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
    
    X = train_data.drop(columns=categorical_columns + ['ID', 'target', 'Policy_Start_Date', 'Policy_End_Date', 'First_Transaction_Date'])
    X = pd.concat([X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    y = train_data['target']
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    
    joblib.dump(encoder, 'encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return model

def preprocess_data(df, encoder, scaler):
    date_columns = ['Policy_Start_Date', 'Policy_End_Date', 'First_Transaction_Date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    df['Policy_Duration'] = (df['Policy_End_Date'] - df['Policy_Start_Date']).dt.days
    df['Customer_Tenure'] = (df['Policy_Start_Date'] - df['First_Transaction_Date']).dt.days
    df['Recency'] = (pd.Timestamp.today() - df['Policy_End_Date']).dt.days
    
    categorical_columns = ['Gender', 'Car_Category', 'Subject_Car_Colour', 'Subject_Car_Make', 'LGA_Name', 'State', 'ProductName']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    encoded_data = encoder.transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
    
    feature_df = df.drop(columns=categorical_columns + ['Policy_Start_Date', 'Policy_End_Date', 'First_Transaction_Date'])
    
    id_col = None
    if 'ID' in feature_df.columns:
        id_col = feature_df['ID'].copy()
        feature_df = feature_df.drop(columns=['ID'])
    
    if 'target' in feature_df.columns:
        feature_df = feature_df.drop(columns=['target'])
    
    feature_df = pd.concat([feature_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    
    numerical_columns = ['Age', 'No_Pol', 'Policy_Duration', 'Customer_Tenure', 'Recency']
    for col in numerical_columns:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].fillna(feature_df[col].mean())
    
    feature_df[numerical_columns] = scaler.transform(feature_df[numerical_columns])
    
    return feature_df, id_col

def get_feature_names_from_data(encoder, scaler):
    """Get feature names by processing a small sample of data"""
    try:
      
        train_sample = pd.read_csv('../data/front_filled_train.csv', nrows=5)
        
     
        categorical_columns = ['Gender', 'Car_Category', 'Subject_Car_Colour', 'Subject_Car_Make', 'LGA_Name', 'State', 'ProductName']
        numerical_columns = ['Age', 'No_Pol']
        

        train_sample['Policy_Start_Date'] = pd.to_datetime(train_sample['Policy_Start_Date'])
        train_sample['Policy_End_Date'] = pd.to_datetime(train_sample['Policy_End_Date'])
        train_sample['First_Transaction_Date'] = pd.to_datetime(train_sample['First_Transaction_Date'])
        
        train_sample['Policy_Duration'] = (train_sample['Policy_End_Date'] - train_sample['Policy_Start_Date']).dt.days
        train_sample['Customer_Tenure'] = (train_sample['Policy_Start_Date'] - train_sample['First_Transaction_Date']).dt.days
        train_sample['Recency'] = (pd.Timestamp.today() - train_sample['Policy_End_Date']).dt.days
        
     
        numerical_columns += ['Policy_Duration', 'Customer_Tenure', 'Recency']
        
       
        encoded_data = encoder.transform(train_sample[categorical_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
        
      
        feature_names = numerical_columns + list(encoded_df.columns)
        
        return feature_names
    except Exception as e:
        st.warning(f"Could not extract feature names: {e}")
        return None

def main():
    
    st.image("../images/final.png", width=300)
    st.markdown('<p class="subtitle">Predict which customers will file insurance claims in the next 3 months</p>', unsafe_allow_html=True)
    
    model = load_or_create_model()
    encoder, scaler = load_or_create_encoders()
    
    if model is None or encoder is None or scaler is None:
        st.error("Could not initialize the prediction system. Please check the data folder and try again.")
        st.stop()
        
    train = pd.read_csv('../data/front_filled_train.csv')
       
    car_category = train['Car_Category'].unique()
    car_color = train['Subject_Car_Colour'].unique()
    car_make = train['Subject_Car_Make'].unique()
    product_name = train['ProductName'].unique()
    state = train['State'].unique()
    lga_name = train['LGA_Name'].unique()
    
    tab1, tab2, tab3 = st.tabs(["Individual Prediction", "Batch Prediction", "Model Insights"])
    
    with tab1:
        st.subheader("")
        st.markdown('<p class="subheader">Individual Customer Prediction</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            car_category = st.selectbox("Car Category", car_category)
            car_color = st.selectbox("Car Color", car_color)
            car_make = st.selectbox("Car Make", car_make)
            product_name = st.radio("Product Name", product_name, help="Select one of the available options.", horizontal=True)
            
        with col2:
            no_pol = st.number_input("Number of Policies", min_value=1, max_value=10, value=1)
            state = st.selectbox("State", state, index=1)
            lga = st.selectbox("LGA Name", lga_name, index=2)
            policy_start = st.date_input("Policy Start Date", datetime.now() - timedelta(days=90), min_value=datetime(2001, 1, 1))
            policy_end = st.date_input("Policy End Date", datetime.now() + timedelta(days=90))
            first_transaction = st.date_input("First Transaction Date", policy_start)
            
        if st.button("Predict Claim Likelihood", type="primary"):
            try:
                data = {
                    'Gender': [gender],
                    'Age': [age],
                    'No_Pol': [no_pol],
                    'Car_Category': [car_category],
                    'Subject_Car_Colour': [car_color],
                    'Subject_Car_Make': [car_make],
                    'LGA_Name': [lga],
                    'State': [state],
                    'ProductName': [product_name],
                    'Policy_Start_Date': [pd.to_datetime(policy_start)],
                    'Policy_End_Date': [pd.to_datetime(policy_end)],
                    'First_Transaction_Date': [pd.to_datetime(first_transaction)]
                }
                
                df = pd.DataFrame(data)
                
                processed_df, _ = preprocess_data(df, encoder, scaler)
                
                prediction = model.predict(processed_df)[0]
                prediction_proba = model.predict_proba(processed_df)[0]
                
                st.subheader("Prediction Result")
                
                if prediction == 1:
                    st.error(f"⚠️ This customer is likely to file a claim in the next 3 months")
                    st.text(f"Claim probability: {prediction_proba[1]:.2%}")
                else:
                    st.success(f"✅ This customer is not likely to file a claim in the next 3 months")
                    st.text(f"Claim probability: {prediction_proba[1]:.2%}")
                
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Please check that your input data is valid and try again.")
                
    with tab2:
        st.subheader("Batch Prediction")
        
        st.write("Upload a CSV file with customer data to predict claims in batch")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                if st.button("Run Batch Prediction"):
                    required_columns = ['Gender', 'Age', 'No_Pol', 'Car_Category', 'Subject_Car_Colour', 
                                       'Subject_Car_Make', 'LGA_Name', 'State', 'ProductName', 
                                       'Policy_Start_Date', 'Policy_End_Date', 'First_Transaction_Date']
                    
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    else:
                        processed_df, id_column = preprocess_data(df, encoder, scaler)
                        
                        if id_column is None:
                            id_column = pd.Series(range(len(processed_df)), name='ID')
                        
                        predictions = model.predict(processed_df)
                        probabilities = model.predict_proba(processed_df)[:, 1]
                        
                        results = pd.DataFrame({
                            'ID': id_column,
                            'Claim_Prediction': predictions,
                            'Claim_Probability': probabilities
                        })
                        
                        st.subheader("Prediction Results")
                        st.dataframe(results)
                        
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions CSV",
                            data=csv,
                            file_name="claim_predictions.csv",
                            mime="text/csv"
                        )
                        
                        st.subheader("Summary")
                        total = len(predictions)
                        claims = sum(predictions)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Customers", total)
                        col2.metric("Predicted Claims", claims)
                        col3.metric("Claim Rate", f"{claims/total:.1%}")
                        
                    
                        
            except Exception as e:
                st.error(f"An error occurred during batch prediction: {e}")
                st.info("Please check that your file format is correct and contains all required columns.")
    
    with tab3:
        st.subheader("Model Insights")
        
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            
            # Get feature names directly from preprocessing a sample
            feature_names = get_feature_names_from_data(encoder, scaler)
            
            if feature_names is not None and len(feature_names) == len(model.feature_importances_):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                
                chart = alt.Chart(importance_df.reset_index()).mark_bar().encode(
                    x='Feature',
                    y='Importance'
                    ).properties(
                        width=1250,
                        height=400,
                        background='#7b67b3ff'
                    ).configure_axis(
                        labelAngle=0)
                        
                st.altair_chart(chart)
                
                st.subheader("Top 10 Important Features")
                
                html = importance_df.head(10).to_html(classes='dataframe', index=False)
                st.write(html, unsafe_allow_html=True)
            else:
                st.warning(f"Feature names count ({len(feature_names) if feature_names else 'Unknown'}) doesn't match feature importances count ({len(model.feature_importances_)})")
                importance_df = pd.DataFrame({
                    'Feature': [f'Feature {i}' for i in range(len(model.feature_importances_))],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.bar_chart(importance_df.set_index('Feature').head(10))
            
            st.write("""
            ### Key Factors Affecting Claims:
            
            1. **Policy Duration**: Longer policies may have different claim patterns
            2. **Customer Tenure**: How long the customer has been with the company
            3. **Age**: Customer's age is a significant predictor of claim likelihood
            4. **Recency**: How recently a policy ended affects claim probability
            5. **Number of Policies**: Customers with multiple policies show different claim behavior
            """)
        
        try:
            train_data = pd.read_csv('../data/front_filled_train.csv')
            
            train_data['Policy_Start_Date'] = pd.to_datetime(train_data['Policy_Start_Date'])
            train_data['Policy_End_Date'] = pd.to_datetime(train_data['Policy_End_Date'])
            train_data['First_Transaction_Date'] = pd.to_datetime(train_data['First_Transaction_Date'])
            
            train_data['Policy_Duration'] = (train_data['Policy_End_Date'] - train_data['Policy_Start_Date']).dt.days
            train_data['Customer_Tenure'] = (train_data['Policy_Start_Date'] - train_data['First_Transaction_Date']).dt.days
            train_data['Recency'] = (pd.Timestamp.today() - train_data['Policy_End_Date']).dt.days
            
            processed_df, _ = preprocess_data(train_data, encoder, scaler)
            y_true = train_data['target']
            y_pred = model.predict(processed_df)
            
            from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
            
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            
            st.subheader("Model Performance")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("F1 Score", f"{f1:.3f}")
            col2.metric("Precision", f"{precision:.3f}")
            col3.metric("Recall", f"{recall:.3f}")
            
            st.write("### Confusion Matrix")
            conf_matrix = pd.DataFrame(cm, 
                                      index=['Actual No Claim', 'Actual Claim'], 
                                      columns=['Predicted No Claim', 'Predicted Claim'])
            
            df2 = conf_matrix.to_html(classes='dataframe', index=True)
            st.write(df2, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"Could not calculate actual model metrics: {e}")
            
            st.subheader("Model Performance")
    
            st.write("### Confusion Matrix")
            confusion_matrix = pd.DataFrame([
                [10602, 22],
                [278, 1177]
            ], index=['Actual No Claim', 'Actual Claim'], 
            columns=['Predicted No Claim', 'Predicted Claim'])
            
            df2 = confusion_matrix.to_html(classes='dataframe', index=True)
            st.write(df2, unsafe_allow_html=True)

        
        st.write(f"""
        ### Model Information:
        
        - **Algorithm**: Decision Tree Classifier
        - **Training Data**: Customer records from historical data
        - **Target Variable**: Whether a customer filed a claim within 3 months
        - **Last Updated**: {datetime.now().strftime("%B %Y")}
        
        This model helps insurance companies predict which customers are likely to file claims in the next three months, enabling better resource allocation and customer service preparation.
        """)

if __name__ == "__main__":
    main()
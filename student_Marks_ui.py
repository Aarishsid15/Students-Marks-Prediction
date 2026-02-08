import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import joblib

# load dataset
df = pd.read_csv('Students Performance Dataset.csv')

# Dividing column into input and ouput
X = df.drop(columns=['Student_ID', 'First_Name', 'Last_Name', 'Email', 'Grade', 'Extracurricular_Activities', 'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level', 'Sleep_Hours_per_Night'], axis=1)
y = df['Total_Score']

#Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)

# Divide data into categorical and numerical
cat_col = ['Gender', 'Department']
num_col = ['Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Study_Hours_per_Week']

#convert text into numerical and numerical into scale value
convert = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), num_col),
        ('cat', OneHotEncoder(handle_unknown= 'ignore'), cat_col)
    ]
)

# using pipeline for automate the workflow
model = Pipeline(steps=[
    ('transform', convert),
    ('linear_regression', LinearRegression())
])

# Training the model
model.fit(X_train, y_train)

#saving the model
joblib.dump(model, 'Marks prediction.pkl')

# testing the model
y_pred = model.predict(X_test)

# creating user input for prediction
# User Interface 
left, center, right = st.columns([1,4,1])
with center:
    st.title('Marks Prediction📑')
st.subheader('Predict your annual exam marks before the result😉')
if st.toggle('Personal Information'):
    with st.container(border=True):
        l,c,r = st.columns([1,4,1])
        with c:
            
            age = st.slider('Age',17,22,1)
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.segmented_control('Gender', ['Male', 'Female'])
        with col2:
            dept = st.segmented_control('Department', ['Mathematics', 'Business', 'Engineering', 'Computer Science'])
        
        co1, co2 = st.columns(2)
        with co1:
            name = st.text_input("Student Name")
        with co2:
            attendance = st.number_input('Attendance %', min_value = 1, max_value = 100)

if st.toggle('College Performance'):
    with st.container(border=True):
        a,b = st.columns(2)
        with a:
            mid_term = st.number_input('Midterm Marks %', min_value = 0, max_value = 100)
        
        with b:
            final_term = st.number_input('Finalterm Marks %', min_value = 0, max_value = 100)
        
        c,d = st.columns(2)
        with c:
            Assign_avg = st.number_input('Aissgnment Avg %', min_value = 0, max_value = 100)
        
        with d:
            Quizzes_avg = st.number_input('Quiz Avg %', min_value=0, max_value = 100)
        
        e,f = st.columns(2)
        with e:
            participation_score = st.number_input("Participation Marks %", min_value=0, max_value=100)
        
        with f:
            project_score = st.number_input('Project Marks %', min_value=0, max_value=100)
        
        x,y,z = st.columns([1,4,1])
        with y:
            study_hours =  st.slider('Study Hours', 1,12,1)
    
    but = st.button('Start')

    #     # converting user input into dataframe.
    user_input = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'Department': dept,
        'Attendance (%)': attendance,
        'Midterm_Score': mid_term,
        'Final_Score': final_term,
        'Assignments_Avg': Assign_avg,
        'Quizzes_Avg': Quizzes_avg,
        'Participation_Score': participation_score,
        'Projects_Score': project_score,
        'Study_Hours_per_Week': study_hours
    }])

    # Predicting the price of house based on user input
    pred_price = model.predict(user_input)[0]
    
    if but:
        if 90 <= pred_price <= 100:
            ll,mm,rr = st.columns([1,4,1])
            with mm:
                with st.container(border=True):
                    st.success(f'{name} Based on your record, the obtained marks are {np.round(pred_price, 2)}%')
                    st.success('Grade: A')
                st.snow()
                with st.expander('Your Following Records'):
                    st.info(f'Gender: {gender}')
                    st.info(f'Age: {age}')
                    st.info(f'Attendance: {attendance}%')
                    st.info(f'Midterm Marks: {mid_term}%')
                    st.info(f'Finalterm Marks: {final_term}%')
                    st.info(f'Quiz Avg: {Quizzes_avg}%')
                    st.info(f'Participation Marks: {participation_score}%') 
                    st.info(f'Project Marks: {project_score}%')
                    st.info(f'Studied hours: {study_hours}')                            

        elif 80 <= pred_price <= 89:
            lm,ml,rm = st.columns([1,4,1])
            with ml:
                with st.container(border=True):
                    st.success(f'{name} Based on your record, the obtained marks are {np.round(pred_price,2)}%')
                    st.success('Grade: B')
                st.snow()
                with st.expander('Your Following Records'):
                    st.info(f'Gender: {gender}')
                    st.info(f'Age: {age}')
                    st.info(f'Attendance: {attendance}%')
                    st.info(f'Midterm Marks: {mid_term}%')
                    st.info(f'Finalterm Marks: {final_term}%')
                    st.info(f'Quiz Avg: {Quizzes_avg}%')
                    st.info(f'Participation Marks: {participation_score}%') 
                    st.info(f'Project Marks: {project_score}%')
                    st.info(f'Studied hours: {study_hours}')

        elif 70 <= pred_price <= 79:
            ri,ce,le = st.columns([1,4,1])
            with ce:
                st.success(f'{name} Based on your record, the obtained marks are {np.round(pred_price,2)}%')
                st.success('Grade: C')
            st.snow()
            with st.expander('Your Following Records'):
                    st.info(f'Gender: {gender}')
                    st.info(f'Age: {age}')
                    st.info(f'Attendance: {attendance}%')
                    st.info(f'Midterm Marks: {mid_term}%')
                    st.info(f'Finalterm Marks: {final_term}%')
                    st.info(f'Quiz Avg: {Quizzes_avg}%')
                    st.info(f'Participation Marks: {participation_score}%') 
                    st.info(f'Project Marks: {project_score}%')
                    st.info(f'Studied hours: {study_hours}')

        elif 60 <= pred_price <= 69:
            aa,ri,sh = st.columns([1,4,1])
            with ri:
                st.success(f'{name} Based on your record, the obtained marks are {np.round(pred_price,2)}%')
                st.success('Grade: D')
            st.snow()
            with st.expander('Your Following Records'):
                    st.info(f'Gender: {gender}')
                    st.info(f'Age: {age}')
                    st.info(f'Attendance: {attendance}%')
                    st.info(f'Midterm Marks: {mid_term}%')
                    st.info(f'Finalterm Marks: {final_term}%')
                    st.info(f'Quiz Avg: {Quizzes_avg}%')
                    st.info(f'Participation Marks: {participation_score}%') 
                    st.info(f'Project Marks: {project_score}%')
                    st.info(f'Studied hours: {study_hours}')

        elif 50 <= pred_price <= 59:
            mo,nt,ty = st.columns([1,4,1])
            with nt:
                st.warning(f'{name} Based on your record, the obtained marks are {np.round(pred_price,2)}%')
                st.warning('Grade: E')
                st.error('Need more attention to Study')
            # st.snow()
            with st.expander('Your Following Records'):
                    st.info(f'Gender: {gender}')
                    st.info(f'Age: {age}')
                    st.info(f'Attendance: {attendance}%')
                    st.info(f'Midterm Marks: {mid_term}%')
                    st.info(f'Finalterm Marks: {final_term}%')
                    st.info(f'Quiz Avg: {Quizzes_avg}%')
                    st.info(f'Participation Marks: {participation_score}%') 
                    st.info(f'Project Marks: {project_score}%')
                    st.info(f'Studied hours: {study_hours}')

        elif 0 <= pred_price <= 49:
            ya,am,mn = st.columns([1,4,1])
            with am:
                st.warning(f'{name} Based on your record, the obtained marks are {np.round(pred_price,2)}%')
                st.warning('Grade: F')
                st.error('Need more attention to Study')
            # st.snow()
            with st.expander('Your Following Records'):
                    st.info(f'Gender: {gender}')
                    st.info(f'Age: {age}')
                    st.info(f'Attendance: {attendance}%')
                    st.info(f'Midterm Marks: {mid_term}%')
                    st.info(f'Finalterm Marks: {final_term}%')
                    st.info(f'Quiz Avg: {Quizzes_avg}%')
                    st.info(f'Participation Marks: {participation_score}%') 
                    st.info(f'Project Marks: {project_score}%')
                    st.info(f'Studied hours: {study_hours}')
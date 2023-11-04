import streamlit as st
import pickle
import time

@st.cache_resource
def model_load(path):
    model = pickle.load(open(path, 'rb'))
    return model

@st.cache_resource
def transformation_load(path):
    transformation = pickle.load(open(path, 'rb'))
    return transformation


scaler = transformation_load('scaler.pkl')
model = model_load('model.pkl')

# Image
st.image('istockphoto-868640146-1024x1024.jpg', width=400)

# Title
st.title('Medical Insurance cost predictor')
st.markdown('#### This model can predict Medical charges with an accuracy score of 90%')

# input field
st.markdown("#### Age")
age = st.text_input('Age: ')

st.markdown('#### Gender')
gender = st.text_input("Male-1, female-0")

st.markdown("#### BMI")
bmi = st.text_input("Enter BMI value in range of (15-55)")

st.markdown("#### Number of Children")
children = st.text_input("Input number")

st.markdown("#### Smoker")
smoker = st.text_input("Smoke: Yes - 1, No - 0")

st.markdown("#### Region")
region = st.text_input("Region: southwest-0, southeast-1, northwest-2, northeast-3: ")

if st.button('Predict'):
    try:
        data = [age, bmi, children, region, gender, smoker]
        scaled_data = scaler.transform([data])
    except ValueError:
        st.markdown("### Please enter valid data !")

    else:
        result = model.predict(scaled_data)

        bar = st.progress(50)
        time.sleep(1)
        bar.progress(100)

        st.info('Success')
        st.markdown(f'**Your Predicted health Insurance charge is : $ {result}**')




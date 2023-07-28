import streamlit as st
import time
import pickle
import numpy as np
with open(r"C:\Users\ASUS\PycharmProjects\Car_Price_predication\transformer.pkl",'rb') as file:
    transfomer=pickle.load(file)
with open(r"C:\Users\ASUS\PycharmProjects\Car_Price_predication\pca.pkl",'rb') as file:
    pca=pickle.load(file)

with open(r"C:\Users\ASUS\PycharmProjects\Car_Price_predication\estimator.pkl",'rb') as file:
    estimator=pickle.load(file)
st.title('Car Price Prediction ')
tab1, tab2, tab3 = st.tabs(["Home", "About", "Programmer"])
with tab1:
    st.image('https://img.etimg.com/photo/45976719/12299004025_e49faafb45_k.jpg')
    col1,col2=st.columns(2)
    with col1:
        year=st.number_input('Enter the year ' ,min_value=2000,step=1)
    with col2:
        km= st.number_input("Enter the kms driven",step=1)
    fuel=st.selectbox('Select the fuel type',('Petrol','Diesel'))
    seller_type = st.selectbox('Select the seller type', ('Individual', 'Dealer','Trustmark Dealer'))
    transmission= st.selectbox('Select the transmission',('Manual','Automatic'))
    owner=st.selectbox('Select the type of Owner',('First Owner','Second Owner','Third Owner','Fouth and Above Owner','Test Drive Car'))
    milege=st.number_input('Enter the milege of the car',min_value=10,step=1)
    engine=st.number_input('Enter the engine(litres)',step=1)
    seats=st.number_input('Enter the number of seats',step=1,min_value=4)
    name=st.selectbox('Select the car brand name',('Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Fiat', 'Datsun', 'Jeep',
       'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW',
       'Nissan', 'MG', 'Daewoo', 'Volvo', 'Kia', 'Jaguar', 'Force',
       'Land', 'Ambassador', 'Ashok', 'Opel', 'Isuzu', 'Peugeot'))
    input_=np.array([year,km,fuel,seller_type,transmission,owner,milege,engine,seats,name],dtype=object).reshape(1,10)
    trans=transfomer.transform(input_).toarray()
    trans=pca.transform(trans)



    on=st.button('check')
    if on:
        out = estimator.predict(trans)

        st.success(f'The predicted price of the car with the given features  is Rs {round(out[0])}')
    with st.sidebar:
       st.text('''This website helps to find price 
of second hand used cars.''')
with tab2:
    st.subheader('Information')
    st.write('This app will help its users to find the possible price of the used second hand cars with the help of given features. The algorithm needs to ask few questions in order to predict the price of a car.')
    st.subheader('Algorithm')
    st.write('''The machine learning model used in this app is Random Forest Regressor. This model is about 82 percent accurate. I have also implemented Grid Search CV in order to get the best parameters for the algorithm.'
Also, used Cross Validation for 20 times to sure about the accuracy of the model. And random forest eliminates the fear of overfitting.''')
with tab3:
    st.subheader('Greetings, dear visitor')
  #  st.image(r"C:\Users\ASUS\Downloads\self.jpg")
    st.write('Myself Soyal Sayyad. Currently I am pursing B.Tech in Computer Science and Engineering at JIT Nagpur')
    st.write('Below is my Linkedin profile link and github link. Do visit and send me an invitation on linkedin.')
    st.write('https://www.linkedin.com/in/soyal10')
    st.write('https://github.com/Soyal10')

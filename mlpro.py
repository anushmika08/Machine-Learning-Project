import streamlit as st
import pickle
from PIL import Image



def main():
    st.title(":rainbow[customer churn prediction]")

    image=Image.open('Churn Prediction with AutoML.jpg')

    st.image(image,width=800)



    

    credit_score=st.text_input("credit_score"," ")


    gender=st.text_input("gender"," ")

    age=st.text_input("age"," ")

    tenure=st.text_input("tenure"," ")

    balance=st.text_input("balance"," ")

    products_number=st.text_input("products_number"," ")

    credit_card=st.text_input("credit_card"," ")

    active_member=st.text_input("active_member"," ")

    estimated_salary=st.text_input("estimated_salary"," ")

    country_France=st.text_input("country_France"," ")

    country_Germany=st.text_input("country_Germany"," ")

    country_Spain=st.text_input("country_Spain"," ")


    features=[[credit_score,gender,age,tenure,balance,products_number,credit_card,active_member,estimated_salary,country_France,country_Germany,country_Spain]]


    model1=pickle.load(open('model1.pkl','rb'))
    sc=pickle.load(open('sc.pkl','rb'))



    pred=st.button("predict")


    if pred:
        feature=sc.transform(features)

        prediction=model1.predict(feature)


        if prediction==1:
            st.write("left the bank")

        else:
            st.write("not left the bank")

main()





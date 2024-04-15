#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from streamlit_extras.switch_page_button import switch_page
# import pickle
from Sentiment_analyser_python import classification
# from Data_Filtered_py import cleaner

# # st.write("---")
# st.markdown(""" <style> .font {
# # font-size:50px ; font-family: 'Cursive'; color: white;text-align:center;} 
# </style> """, unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;colour:white;'>Sentiment Analyser</h1>",unsafe_allow_html=True)
st.write("---")
st.write("<p style='text-align:center;color:white;'>Sentiment analysis looks at the emotion expressed in a text. It is commonly used to analyze customer feedback, survey responses, and product reviews.</p>",unsafe_allow_html=True)
st.write("###")
# upload_file=st.file_uploader("Type your Text here",type="text")
title = st.text_input(
    "Input:",
    placeholder="Eg.It's a great product and has a great design!"
)
# st.write(model)
# st.write(clean)
if title:
    res=classification(title)
    if(res[0]==0):
        st.write("<h4 style='color:red;font-weight:bold;'>It's a negative one!</h4>",unsafe_allow_html=True)
    elif(res[0]==1):
        st.write("<h4 style='color:green;font-weight:bold;'>It's a positive one!</h4>",unsafe_allow_html=True)

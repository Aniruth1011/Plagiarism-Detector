# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:00:58 2023

@author: Aniruth
"""
import plagiarism_detector as detect
import streamlit as st
source = ""
check = ""
# Define the pages
def home():
    st.title("Submit Paragraphs")
    source = st.text_area("Enter Source  text")
    if source is not None:
        srce = source.splitlines()
    check = st.text_area("Enter Checking  text")
    if check is not None:
        chk = check.splitlines()
    source = srce
    check = chk
    if st.button('Submit'):
        st.title("Plagiarism Detection Page")
        source_sentence , check_sentence = detect.solution(str(source) , str(check))
        for i in range(len(source_sentence)):
            st.write(i+1 , "Source Sentence Detected : ")
            st.write(str(source_sentence[i]))
            st.write("\n")
            st.write("Sentence in Checking Paragraph ")
            st.write(str(check_sentence[i]))
home()

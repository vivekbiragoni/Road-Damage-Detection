import streamlit as st
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

# data=[]
# df= pd.DataFrame(data)
# x=len(df[df['Num Objects']==0])

st.title('Automated Road Damage Detection')
st.text("       ")
st.text("       ")
st.text("       ")
st.text("       ")

def explore():
      st.header('Analysis of trained road data' )
      st.text("       ")
      st.subheader('1. Distribution of roads in trained data')
      st.image('plt1.png',width=400)
      with st.expander('See explanation'):
            code="""
                  labels = ['good roads', 'damaged roads']
                  plt.pie([x,7706-x],  labels=labels,  autopct='%0.2f')
                  plt.title('Distribution of roads in train data') """
            st.code(code,language='python')

            st.write('''It is a pie chart showing distribution between good and bad roads in train data
                      we can infer that 50.88 percent are good and 49.12 are damaged roads with atleast 1 crack ''')

      st.subheader('2. No of Cracks in each type in train data')
      st.image('plt2.png',width=400)
      with st.expander('See explanation'):
            code="""
                  plt.bar(x=['Longitudinal ','Transverse','Alligator ','Pothole','other'],height=[d0,d1,d2,d4,d5])
                  plt.title('No of cracks in each type in train data')
                  plt.xlabel('Type of crack')
                  plt.ylabel('Total number of cracks') """
            st.code(code, language='python')
            st.write('This is a bar graph explaining no. of different types of cracks in the trained data. By the plot it is clear that potholes are in large number when compared to other cracks')

      st.subheader('3. Percentage of cracks in each type in train data')
      st.image('plt3.png',width=400)
      with st.expander('See explanation'):
            code="""
                  plt.pie([d0,d1,d2,d4,d5],  labels=['Longitudinal ','Transverse','Alligator ','Pothole','other'],  autopct='%0.2f')
                  plt.title('percent of cracks in each type in train data') """
            st.code(code, language='python')
            st.write('This is a pie chart explaining no. of different types of cracks(in percentage) in the trained data. By the plot it is clear that potholes are in large number when compared to other cracks')

      st.subheader('4. Maximum no. of cracks in different types in train data')
      st.image('plt4.png',width=400)
      with st.expander('See explanation'):
            code="""
                  plt.bar(x=['Longitudinal ','Transverse','Alligator ','Pothole','other'],height=[df['D00'].max(),df['D10'].max(),df['D20'].max(),df['D40'].max(),df['other'].max()])
                  plt.title('maximum of cracks in each type in train data')
                  plt.xlabel('Type of crack')
                  plt.ylabel('Maximum number of cracks') """
            st.code(code, language='python')
            st.write('This is a bar graph explaining maximum no. of cracks in different types of cracks in the trained data.')

      st.subheader('5. Image IDs with maximum cracks')
      st.image('plt5.png',width=400)
      with st.expander('See explanation'):
            code="""
                  df['Num Objects'].sort_values(ascending=False).head().plot(kind='bar')
                  plt.title('Image ids in train data with maximum cracks')
                  plt.xlabel('image id')
                  plt.ylabel('maximum crack') """
            st.code(code, language='python')
            st.write('This is a bar graph explaining  the image ids with maximum no. of cracks in trained data.')

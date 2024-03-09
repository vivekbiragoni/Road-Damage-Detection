import streamlit as st
import pandas as pd
from PIL import Image
from road import road
from info import explore

page = st.sidebar.selectbox("Explore Or Inspect road", ("Inspect road", "Explore"))

if page=='Inspect road':
      road()
elif page=='Explore':
      explore()




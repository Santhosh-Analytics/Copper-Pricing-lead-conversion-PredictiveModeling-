import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
# from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_option_menu import option_menu

# image='ss.jpg'

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Copper Modeling",
    page_icon='https://www.shirepost.com/cdn/shop/files/WOR-HAM-RAW-CO-1.jpg?v=1710180524&width=1800',
    )

with st.sidebar:
    st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)
    
    selected = option_menu("Main Menu", ["About", 'Predictor'], 
        icons=['house-door-fill','bar-chart-fill'], menu_icon="cast", default_index=0,styles={
        "container": {"padding": "0!important", "background-color": "gray"},
        "icon": {"color": "rgb(235, 48, 84)", "font-size": "25px","font-family":"inherit"}, 
        "nav-link": {"font-family":"inherit","font-size": "22px", "color": "#ffffff","text-align": "left", "margin":"0px", "--hover-color": "#84706E"},
        "nav-link-selected": {"font-family":"inherit","background-color": "azure","color": "#FF385C","font-size": "25px"},
    })
    st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

st.markdown(""" <style> button[data-baseweb="tab"] > di v[data-testid="stMarkdownContainer"] > p {font-size: 28px;} </style>""", unsafe_allow_html=True)
st.markdown('<style>div.css-1jpvgo6 {font-size: 16px; font-weight: bolder;font-family:inherit; } </style>', unsafe_allow_html=True)



st.markdown("<h1 style='text-align: center; font-size: 38px; color: #FF385C ; font-weight: 700;font-family:inherit;'>Streamline Your Copper Business: Pricing & Lead Management</h1>", unsafe_allow_html=True)


st.markdown("<hr style='border: 2px solid beige;'>", unsafe_allow_html=True)

if selected == "About":
  
  st.markdown("<h3 style='font-size: 30px;text-align:left; font-family:inherit;color: #FF385C;'> Overview  </h3>", unsafe_allow_html=True)

  st.markdown("""<p  style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400;font-family:inherit;'>  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      
                      This project aims to develop machine learning models to address the challenges faced by the copper industry in pricing and lead classification. By leveraging advanced techniques such as data normalization, feature scaling, and outlier detection, the project delivers robust solutions for accurate pricing decisions and effective lead classification.
              </p>""", unsafe_allow_html=True)

  st.markdown("<h3 style='font-size: 30px;text-align:left; font-family:inherit;color: #FF385C;'> Models  </h3>", unsafe_allow_html=True)

  st.markdown("""<p  style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400;font-family:inherit;'>  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      
                    Regression model: XG Boost Regressor  for predicting the continuous variable 'Selling_Price'. Classification model: XG Boost Classifier for predicting lead status (WON or LOST).
            </p>""", unsafe_allow_html=True)

  st.markdown("<h3 style='font-size: 30px;text-align:left; font-family:inherit;color: #FF385C;'> Contributing  </h3>", unsafe_allow_html=True)
  github_url = "https://github.com/Santhosh-Analytics/Copper_Modeling"
  st.markdown("""<p  style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400;font-family:inherit;'>  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      
                    Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request in the <a href="{}">GitHub Repository</a>.
            </p>""".format(github_url)
, unsafe_allow_html=True)



if selected == "Predictor":

  tab, tab1= st.tabs(["***PricePrediction***","***Lead Classification***"])
  
  with tab:
    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
  '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
  '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
  '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
  '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
  st.selectbox('Select Status',status_options)
  st.selectbox('Select Item Type',item_type_options)
  st.selectbox('Select Country Code',country_options)
  st.selectbox('Select Application',application_options)
  st.selectbox('Select Product',product)

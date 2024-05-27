import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from scipy import stats
from scipy.stats import boxcox
import pickle
from datetime import date, timedelta
from streamlit_option_menu import option_menu

# Function to perform Box-Cox transformation on a single value using a given lambda
def transform_single_value(value, lmbda):
    if value is None:
        return None  # Handle missing value
    transformed_value = boxcox([value], lmbda=lmbda)[0]
    return transformed_value

# Load the saved lambda values
with open(r'pkls/boxcox_lambdas.pkl', 'rb') as f:
    lambda_dict = pickle.load(f)

# Load other required models and encoders
with open(r'pkls/Class_ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)

with open(r'pkls/scale_class.pkl', 'rb') as f:
    scale_class = pickle.load(f)

with open(r'pkls/xgb_classifier_model.pkl', 'rb') as f:
    xgb_classifier_model = pickle.load(f)
    
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Copper Modeling",
    page_icon='https://www.shirepost.com/cdn/shop/files/WOR-HAM-RAW-CO-1.jpg?v=1710180524&width=1800',
)

with st.sidebar:
    st.markdown("<hr style='border: 2px solid #ffef96;'>", unsafe_allow_html=True)
    
    selected = option_menu(
        "Main Menu", ["About", 'Predictor'], 
        icons=['house-door-fill', 'bar-chart-fill'], 
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "gray"},
            "icon": {"color": "rgb(235, 48, 84)", "font-size": "25px", "font-family": "inherit"},
            "nav-link": {"font-family": "inherit", "font-size": "22px", "color": "#ffffff", "text-align": "left", "margin": "0px", "--hover-color": "#84706E"},
            "nav-link-selected": {"font-family": "inherit", "background-color": "azure", "color": "#FF385C", "font-size": "25px"},
        }
    )
    st.markdown("<hr style='border: 2px solid #ffef96;'>", unsafe_allow_html=True)

# st.markdown(""" <style> button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {font-size: 32px;} </style>""", unsafe_allow_html=True)
# st.markdown('<style>div.css-1jpvgo6 {font-size: 18px; font-weight: bolder; font-family: inherit;} </style>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 38px; color: #FF385C; font-weight: 700; font-family: inherit;'>Streamline Your Copper Business: Pricing & Lead Management</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid beige;'>", unsafe_allow_html=True)

if selected == "About":
    st.markdown("<h3 style='font-size: 30px; text-align: left; font-family: inherit; color: #FBBC05;'> Overview </h3>", unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400; font-family: inherit;'>
        This project aims to develop machine learning models to address the challenges faced by the copper industry in pricing and lead classification. By leveraging advanced techniques such as data normalization, feature scaling, and outlier detection, the project delivers robust solutions for accurate pricing decisions and effective lead classification.
    </p>""", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 30px; text-align: left; font-family: inherit; color: #FBBC05;'> Models </h3>", unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400; font-family: inherit;'>
        Regression model: XG Boost Regressor for predicting the continuous variable 'Selling_Price'. Classification model: XG Boost Classifier for predicting lead status (WON or LOST).
    </p>""", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 30px; text-align: left; font-family: inherit; color: #FBBC05;'> Contributing </h3>", unsafe_allow_html=True)
    github_url = "https://github.com/Santhosh-Analytics/Copper_Modeling"
    st.markdown("""<p style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400; font-family: inherit;'>
        Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request in the <a href="{}">GitHub Repository</a>.
    </p>""".format(github_url), unsafe_allow_html=True)

if selected == "Predictor":
    tab, tab1 = st.tabs(["***Price Prediction***", "***Lead Classification***"])
    
    with tab:
        # Options for various dropdowns
        item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
        product_options = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                           '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                           '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                           '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                           '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

        # Define the date ranges
        today = date.today()
        six_months_ago = today - timedelta(days=365 // 2)
        six_months_after = today + timedelta(days=180 // 2)
        del_min = today + timedelta(days=2)

        col1, col2 = st.columns(2)

        with col1:
            item_date = st.date_input('Item Date', min_value=six_months_ago, max_value=today, help='Enter product/order date, must be today or within the past six months', value=None)
            del_date = st.date_input('Delivery Date', min_value=del_min, max_value=six_months_after, help='Enter product/order date, must be today or within the next six months', value=None)
            item = st.selectbox('Select Item Type', item_type_options, index=None, help="Select Item type from the dropdown menu", placeholder="Select Item type from the dropdown menu")
            country = st.selectbox('Select Country Code', country_options, index=None, placeholder="Select Country Code from the list")
            app = st.selectbox('Select Application', application_options)
            

        with col2:
          prod = st.selectbox('Select Product', product_options)
            # cust = st.number_input('Customer Number', min_value=30000000, max_value=39999999, value=None, help='Enter Customer number. Min value is 30000000 and max value is 39999999', placeholder='Enter Customer number. Min value is 30000000 and max value is 39999999')
          thick = st.number_input('Thickness', min_value=0.10, max_value=7.0, value=None, help='Enter the thickness of the product. Min value is 0.10 and max value is 7.', placeholder='Enter Thickness of the product. Min value is 0.10 and max value is 7.')
          width = st.number_input('Width', min_value=500, max_value=2000, value=None, help='Enter the width of the product. Min value is 500 and max value is 2000', placeholder='Enter width of the product. Min value is 0.10 and max value is 7.')
          qty = st.number_input('Quantity in tons', min_value=1.00, max_value=800.00, value=None, help='Enter the quantity in tons. Min value is 1.00 and max value is 2.00', placeholder='Enter the quantity in tons. Min value is 1.00 and max value is 2.00')
          # price = st.number_input('Selling price in $', min_value=10.00, value=None, help='Enter selling price in $. Min value is 10.00.', placeholder='Enter selling price in $. Min value is 10.00.')
          st.write(' ')
          st.write(' ')
          button = st.button('Predict Lead')

        volume = None
        # unit_price = None
        days = None
        Item_transform = None

        if qty is not None and width is not None and thick is not None:
            volume = float(qty) * float(thick) * float(width)
            # unit_price = float(price) * float(qty) * float(thick)
            days = (del_date - item_date).days
            Item_transform = ohe.transform([[item]])
        else:
            st.error('Please fill in all the required fields.')

        if None not in (qty, thick, width,  volume):
            qty_box = transform_single_value(qty, lambda_dict['quantity_tons'])
            thick_box = transform_single_value(thick, lambda_dict['thickness'])
            width_box = transform_single_value(width, lambda_dict['width'])
            volume_box = transform_single_value(volume, lambda_dict['volume'])
            # unit_price_box = transform_single_value(unit_price, lambda_dict['unit_price'])
            # price_box = transform_single_value(price, lambda_dict['selling_price'])

            data = np.array([[int(country), int(app), int(prod), int(days), float(qty_box), float(thick_box), float(width_box), float(volume_box)]])
            st.write(item,'\n\n',Item_transform)
            data = np.concatenate((data, Item_transform), axis=1)
            # # indices_to_remove = [11, 15]  # Remove unnecessary indices
            # # data = np.delete(data, indices_to_remove, axis=1)
            # data = data.reshape(1, -1)

            scaled_data = scale_class.transform(data)
            st.write(scaled_data)

            if button:
                prediction = xgb_classifier_model.predict(scaled_data)
                st.write(prediction)
                if prediction == 1:
                    st.success('Won')
                elif prediction==0:
                    st.warning('Lost')
        else:
            st.warning('Please ensure all input fields are filled correctly.')

    with tab1:
        st.info('gvhgvg')

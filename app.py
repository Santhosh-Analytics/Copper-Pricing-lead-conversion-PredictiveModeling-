import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from scipy import stats
from scipy.stats import boxcox
import pickle
from datetime import date, timedelta
from streamlit_option_menu import option_menu
from scipy.special import inv_boxcox



# Set page configuration
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Copper Modeling",
    page_icon='https://www.shirepost.com/cdn/shop/files/WOR-HAM-RAW-CO-1.jpg?v=1710180524&width=1800',
)


# Injecting CSS for custom styling
st.markdown("""
    <style>
    /* Tabs */
    div.stTabs [data-baseweb="tab-list"] button {
        font-size: 25px;
        color: #ffffff;
        background-color: #4CAF50;
        padding: 10px 20px;
        margin: 10px 2px;
        border-radius: 10px;
    }
    div.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #009688;
        color: white;
    }
    div.stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #3e8e41;
        color: white;
    }
    /* Button */
    .stButton>button {
        font-size: 22px;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 16px;
    }
    .stButton>button:hover {
        background-color: #3e8e41;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to perform Box-Cox transformation on a single value using a given lambda
def transform_single_value(value, lmbda):
    if value is None:
        return None  # Handle missing value
    transformed_value = boxcox([value], lmbda=lmbda)[0]
    return transformed_value

def reverse_boxcox_transform(predicted, lambda_val):
    return inv_boxcox(predicted, lambda_val)

# Load the saved lambda values
with open(r'pkls/boxcox_lambdas.pkl', 'rb') as f:
    lambda_dict = pickle.load(f)

# Load other required models and encoders
with open(r'pkls/Class_ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)

with open(r'pkls/robust_scaler.pkl', 'rb') as f:
    robust_scaler = pickle.load(f)

with open(r'pkls/xgb_classifier_model.pkl', 'rb') as f:
    xgb_classifier_model = pickle.load(f)
    
with open(r'pkls/scale_class.pkl', 'rb') as f:
    scale_class = pickle.load(f)
    
with open(r'pkls/Reg_ohe.pkl', 'rb') as f:
    Reg_ohe = pickle.load(f)
    
with open(r'pkls/scale_reg.pkl', 'rb') as f:
    scale_reg = pickle.load(f)

with open(r'pkls/xgb_regression_model.pkl', 'rb') as f:
    xgb_Reg = pickle.load(f)
    


with st.sidebar:
    st.markdown("<hr style='border: 2px solid #ffffff;'>", unsafe_allow_html=True)
    
    selected = option_menu(
        "Main Menu", ["About", 'Predictor'], 
        icons=['house-door-fill', 'bar-chart-fill'], 
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "gray"},
            "icon": {"color": "#000000", "font-size": "25px", "font-family": "Times New Roman"},
            "nav-link": {"font-family": "inherit", "font-size": "22px", "color": "#ffffff", "text-align": "left", "margin": "0px", "--hover-color": "#84706E"},
            "nav-link-selected": {"font-family": "inherit", "background-color": "#ffffff", "color": "#55ACEE", "font-size": "25px"},
        }
    )
    st.markdown("<hr style='border: 2px solid #ffffff;'>", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; font-size: 38px; color: #55ACEE; font-weight: 700; font-family: inherit;'>Streamline Your Copper Business: Pricing & Lead Management</h1>", unsafe_allow_html=True)
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
    tab, tab1 = st.tabs(["***Lead Classification***", "***Price Prediction***"])

    with tab:
    
        # Options for various dropdowns
        item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        country_options = [78, 26, 25, 32, 27, 28, 84, 77, 30, 39, 79, 38, 40, 80, 113, 89, 107]
        application_options =[10,41,15,59,42,56,29,26,27,28,25,40,79,22,66,3,20,38,58,4,65,39,68,67,19,99,5,69,70,2]
        product_options =[611993,164141591,640665,1670798778,628377,1668701718,640405,1671863738,1332077137,1693867550,1668701376,1671876026,628117,164337175,1668701698,1693867563,1721130331,1282007633,628112,1665572374,1690738206,1722207579,611728,640400,611733,1668701725,164336407,1690738219,1665584320,1665572032,1665584642,929423819]

        # Define the date ranges
        today = date.today()
        six_months_ago = today - timedelta(days=365 // 2)
        six_months_after = today + timedelta(days=180 // 2)
        del_min = today + timedelta(days=2)

        col1, col, col2 = st.columns([2,.5,2])

        with col1:
            cust = st.number_input('Customer Number', min_value=30000000, max_value=39999999, value=None, help='Enter Customer number. Min value is 30000000 and max value is 39999999', placeholder='Enter Customer number. Min value is 30000000 and max value is 39999999')
            item_date = st.date_input('Item Date',   help='Enter product/order date, must be today or within the past six months', value=None)
            del_date = st.date_input('Delivery Date',  help='Enter product/order date, must be today or within the next six months', value=None)
            item = st.selectbox('Select Item Type', item_type_options, index=None, help="Select Item type from the dropdown menu", placeholder="Select Item type from the dropdown menu")
            country = st.selectbox('Select Country Code', country_options, index=None, placeholder="Select Country Code from the list")
            app = st.selectbox('Select Application', application_options,index=None, help=' Select from the dropdown' ,placeholder='Select from the dropdown')
            

        with col2:
            prod = st.selectbox('Select Product', product_options,index=None, help=' Select from the dropdown' ,placeholder='Select from the dropdown')
            thick = st.number_input('Thickness', min_value=0.10, max_value=7.0, value=None, help='Min value is 0.10 and max value is 7.', placeholder='Min value is 0.10 and max value is 7.')
            width = st.number_input('Width', min_value=500, max_value=2000, value=None, help=' Min value is 500 and max value is 2000', placeholder='Min value is 500 and max value is 2000')
            qty = st.number_input('Quantity in tons', min_value=1.00, max_value=800.00, value=None, help=' Min value is 1.00 and max value is 800.00', placeholder='Min value is 1.00 and max value is 800.00')
            price = st.number_input('Selling price in $', min_value=10.00, value=None, help='Enter selling price in $. Min value is 10.00.', placeholder='Enter selling price in $. Min value is 10.00.')
            
            st.write(' ')
            st.write(' ')
            button = st.button('Predict Lead')
            
        if None not in (item_date,del_date):
            days = (del_date - item_date).days
            
        del_month_class = del_date.month if del_date is not None  else None
        del_year_class = del_date.year if del_date is not None  else None
        item_year_class = item_date.year if item_date is not None  else None
        item_month_class = item_date.month if item_date is not None  else None
        item_day_class = item_date.day if item_date is not None  else None
        
        # volume = None
        # unit_price = None
        # Item_transform = None
        # qty_box = None
        # scaled_data=None
        # thick_box = None
        # width_box = None 
        # volume_box = None 
        # unit_price_box = None
        # data=None
        # days = None



        
        volume = float(qty) * float(thick) * float(width) if (qty is not None and thick is not None and width is not None) else None
        unit_price = float(price) * float(qty) * float(thick) if (price is not None and qty is not None and thick is not None) else None


        
    
        qty_box = transform_single_value(qty, lambda_dict['quantity_tons'])     if qty is not None  else None    
        thick_box = transform_single_value(thick, lambda_dict['thickness'])      if thick is not None  else None    
        width_box = transform_single_value(width, lambda_dict['width'])        if width is not None  else None    
        volume_box = transform_single_value(volume, lambda_dict['volume']) if volume is not None  else None    
        unit_price_box = transform_single_value(unit_price, lambda_dict['unit_price'])   if unit_price is not None  else None    
        price_box = transform_single_value(price, lambda_dict['selling_price']) if volume is not None and unit_price is not None else None

        
        data = np.array([[cust,country, app, prod, days,del_month_class,del_year_class,item_day_class,item_month_class,item_year_class,qty_box, thick_box,width_box,price_box, volume_box,unit_price_box]])
        st.write(data)
        
        Item_transform = ohe.transform([[item]]) if item is not None else None

        st.write(item,'\n\n',Item_transform)
        
        
        data_final = np.append(data, Item_transform, axis=1) if data is not None  else None
        st.write(data_final)
        
        # reg_scaled_data = robust_scaler.transform(reg_data)  if reg_data is not None  else None
        # # st.write(reg_scaled_data)
        
        # # data = np.concatenate((data,Item_transform) , axis=1) if data is not None else None
        
        # scaled_data = robust_scaler.transform(data)
        # st.write(scaled_data)
            

    if button and data_final is not None:
        
        scaled_data = scale_class.transform(data_final)
        st.write(scaled_data)
        prediction = xgb_classifier_model.predict(scaled_data)
        st.write(prediction)
        if prediction == 1:
            st.snow()
            st.success('Won')
        elif prediction==0:
            st.warning('Lost')
        elif data is None:
            st.error('Update all the fields and hit the button')
        else:
            st.warning('Please ensure all input fields are filled correctly.')

    with tab1:
        
        status_options = ['Won', 'Lost', 'Not lost for AM', 'Revised', 'To be approved', 'Draft', 'Offered',  'Offerable',  'Wonderful']
        
        reg_col1, reg_col2 = st.columns(2)

        with reg_col1:
            reg_item_date = st.date_input('Item Date',  help='Enter product/order date', value=None,key='item_dt')
            reg_del_date = st.date_input('Delivery Date',  help='Enter delivery date', value=None,key='del_dt')
            reg_item = st.selectbox('Select Item Type', item_type_options, index=None, help="Select Item type from the dropdown menu", placeholder="Select Item type from the dropdown menu",key='item')
            reg_country = st.selectbox('Select Country Code', country_options, index=None, placeholder="Select Country Code from the list",key='ctry')
            reg_app = st.selectbox('Select Application', application_options,index=None, help=' Select from the dropdown' ,placeholder='Select from the dropdown',key='app')
            reg_cust = st.number_input('Customer Number', min_value=30000000, max_value=39999999, value=None, help='Enter Customer number. Min value is 30000000 and max value is 39999999', placeholder='Enter Customer number. Min value is 30000000 and max value is 39999999',key='cust')
            

        with reg_col2:
            
            reg_prod = st.selectbox('Select Product', product_options,index=None, help=' Select from the dropdown' ,placeholder='Select from the dropdown',key='prod')
            reg_thick = st.number_input('Thickness', min_value=0.10, max_value=7.0, value=None, help='Min value is 0.10 and max value is 7.', placeholder='Min value is 0.10 and max value is 7.',key='thick')
            reg_width = st.number_input('Width', min_value=500, max_value=2000, value=None, help=' Min value is 500 and max value is 2000', placeholder='Min value is 500 and max value is 2000',key='width')
            reg_qty = st.number_input('Quantity in tons', min_value=1.00, max_value=800.00, value=None, help=' Min value is 1.00 and max value is 800.00', placeholder='Min value is 1.00 and max value is 800.00',key='qty')
            status = st.selectbox('Select Status', status_options,index=None, help=' Select from the dropdown' ,placeholder='Select from the dropdown',key='status')
            
            all_valid = all(var is not None for var in (reg_item_date, reg_del_date, reg_item, reg_country, reg_app, reg_cust, reg_prod, reg_thick, reg_width, reg_qty, status))
            st.write('')
            st.write(' ')
            button = st.button('Predict Price',key='Reg',) if all_valid else st.warning('Please ensure all input fields are filled correctly.')

          
        item_type_mapping = {'WI': 0, 'PL': 1, 'Others': 2, 'IPL': 3, 'S': 4, 'W': 5, 'SLAWR': 7}
        app_mapping = {58: 0, 68: 2, 28: 3, 56: 4, 25: 5, 59: 6, 69: 7, 15: 8, 39: 9, 4: 10, 3: 11, 5: 12, 22: 13, 40: 14, 10: 15, 26: 16, 66: 17, 65: 18, 70: 19, 67: 20, 27: 21, 20: 22, 19: 23, 29: 24, 79: 25, 2: 26, 42: 27, 41: 28, 38: 29, 99: 32}

        item_type_label  = None
        app_label = None
        
        item_type_label = item_type_mapping[reg_item] if reg_item in item_type_mapping else item_type_label
        app_label = app_mapping[reg_app] if reg_app in app_mapping else app_label


        del_year = reg_del_date.year if reg_del_date is not None  else None
        del_month = reg_del_date.month if reg_del_date is not None  else None
        del_time = abs((reg_del_date-reg_item_date).days) if reg_del_date is not None  else None
        status_transform = Reg_ohe.transform([[status]]) if status is not None else None
        reg_volume = float(reg_qty) * float(reg_thick) * float(reg_width) if (reg_qty is not None and reg_thick is not None and reg_width is not None) else None

        


        volume_box = transform_single_value(reg_volume, lambda_dict['volume'])  if reg_volume is not None  else None
        thick_box = transform_single_value(reg_thick, lambda_dict['thickness']) if reg_thick is not None  else None
        width_box = transform_single_value(reg_width, lambda_dict['width']) if reg_width is not None  else None
        qty_box = transform_single_value(reg_qty, lambda_dict['quantity_tons']) if reg_qty is not None  else None
        item_year =reg_item_date.year if reg_item_date is not None  else None
        item_month =reg_item_date.month if reg_item_date is not None  else None
        item_day =reg_item_date.day if reg_item_date is not None  else None


        reg_data = np.array([[reg_cust, reg_country, item_type_label,app_label, reg_prod, del_time,del_month,del_year, item_day,item_month,item_year, qty_box,thick_box, width_box, volume_box]]) if None not in (reg_cust, reg_country, reg_app, reg_prod, del_year, item_month, thick_box, width_box, volume_box) else None


        
        if status is not None:
            ohe_data = Reg_ohe.transform([[status]]) 
        
        # st.write(status,'\n\n',ohe_data,'\n\n',reg_data)
        # st.write(reg_data.shape)
        # st.write(ohe_data.shape)

        reg_data = np.append(reg_data, ohe_data, axis=1) if reg_data is not None  else None
        # st.write(reg_data)
        
        reg_scaled_data = robust_scaler.transform(reg_data)  if reg_data is not None  else None
        # st.write(reg_scaled_data)
        

        
        
        # st.write(reg_scaled_data)
    

        if button and reg_data is not None:
            st.snow()
            prediction = xgb_Reg.predict(reg_scaled_data) 
            # st.write(prediction)
            lambda_val = lambda_dict['selling_price'] 
            transformed_predict=reverse_boxcox_transform(prediction, lambda_val) if reg_data is not None else None
            rounded_prediction = round(transformed_predict[0],2)
            st.success(f'Based on the input, the predicted price is,  {rounded_prediction:.2f}')
            st.snow()
                
        else:
            rounded_prediction = 0
            # st.warning('Please ensure all input fields are filled correctly.')
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_option_menu import option_menu


def st_app_layout(page_title, page_icon=None, markdown_color="#FF385C",
                     menu_options=["Home"], menu_icons=[], menu_icon="cast",
                     default_index=0, menu_styles={}):
    st.set_page_config(
      layout="wide",
      initial_sidebar_state="expanded",
      page_title=page_title,
      page_icon=page_icon,
  )

    with st.sidebar:
      st.markdown(f"<hr style='border: 2px solid {markdown_color};'>", unsafe_allow_html=True)

      selected = option_menu(
          "Main Menu", menu_options, icons=menu_icons, menu_icon=menu_icon,
          default_index=default_index, styles=menu_styles
      )

      st.markdown(f"<hr style='border: 2px solid {markdown_color};'>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 38px; color: #FF385C ; font-weight: 700;font-family:inherit;'>Copper industry Pricing & Lead predictor</h1>", unsafe_allow_html=True)
    st.markdown(f"<hr style='border: 2px solid beige{markdown_color};'>", unsafe_allow_html=True)

page_title = "Predictor"   
menu_options = ["Home", "Price", "Class"]
menu_icons = ["house-door-fill",  "credit-card","bar-chart-fill"]
menu_styles = {
    "container": {"padding": "0!important", "background-color": "gray"},
    "icon": {"color": "rgb(235, 48, 84)", "font-size": "25px", "font-family": "inherit"},
    "nav-link": {"font-family": "inherit", "font-size": "22px", "color": "#ffffff",
                 "text-align": "left", "margin": "0px", "--hover-color": "#84706E"},
    "nav-link-selected": {"font-family": "inherit", "background-color": "azure",
                          "color": "#FF385C", "font-size": "25px"},
}

st_app_layout(page_title, page_icon="path/to/icon.png", menu_options=menu_options,
                  menu_icons=menu_icons, menu_styles=menu_styles)

st.info('Almost completed')
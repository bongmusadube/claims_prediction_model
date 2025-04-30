import streamlit as st

def custom_styling():
    st.markdown(
        """
        <style>
            .main {
                background-image: linear-gradient(to bottom right, #1f1247ff, #7b67b3ff);
            }

            .title {
                font-family: "Open Sans", sans-serif;
                font-size: 50px;
                font-weight: bold;
                color: black;
            }
            
            .subtitle {
                font-family: "Open Sans", sans-serif;
                font-weight: 300;
                font-size: 35px;
                color: white;
                margin-top: -40px
            }
                
            .subheader {
                font-family: "Open Sans", sans-serif;
                font-size: 25px;
                color: white;
                margin-top: -50px
            }

            .text {
                font-family: "Open Sans", sans-serif;
                font-size: 16px;
                color: white;
                max-width: 500px;
            }
            
            .stTabs [data-baseweb="tab"] {
                font-family: "Open Sans", sans-serif;
                font-size: 16px;
                color:  white;
                position : relative
            }
            
            .stTabs [data-baseweb="tab"]:hover {
                font-size: 30px;
                color: #77be5cff;
            }
            
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                font-weight: bold;
                color: #77be5cff;
            }

            .stButton > button {
                font-family: "Open Sans", sans-serif;
                background-color: #1f1247ff;
                color: white;
                border: none;
                padding: 8px 25px;
                border-radius: 25px;
                font-size: 16px;
                font-weight: bold;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                cursor: pointer;
                transition: background-color 0.3s;
            }

            .stButton > button:hover {
                background-color: #77be5cff;
                color: white;
            }
            
            .stSelectbox div[data-baseweb="select"] > div {
                background-color: #77be5cff;
            }
            
            .stSelectbox div[data-baseweb="select"] .css-1uccc91-singleValue {
                background-color: #77be5cff;
            }
            
            .stSelectbox div[data-baseweb="select"] .css-1n7v3ny-option {
                background-color: #77be5cff;
            }
            
            .stNumberInput input {
                background-color: #77be5cff;
            }
            
            .stNumberInput input:focus {
                background-color: #77be5cff;
            }
            
            .stNumberInput > div > div > input[type=number] {
                border-radius: 0;
                padding: 0;
                margin: 0;
            }
            
            .stNumberInput > div > div > button:first-of-type {
                background-color: #77be5cff;
            }
            
            .stNumberInput > div > div > button:last-of-type {
                background-color: #77be5cff;
            }

            .stDateInput input {
                background-color: #77be5cff;
            }
            
            .stDateInput input:focus {
                background-color: #77be5cff;
            }
            
            div[data-baseweb="radio"] > div {
                background-color: white;
                border: 1px solid #000;
                border-radius: 50%;
                width: 16px;
                height: 16px;
                display: inline-block;
                position: relative;
            }
    
            /* Input field styling */
            .stTextInput > div > div > input {
                background-color: white;
                color: #333;
                border-radius: 5px;
                border: 1px solid #ddd;
                padding: 8px 12px;
            }

            .stTextInput > div > div > input:focus {
                border-color: #FFC107;
                box-shadow: 0 0 0 2px rgba(255, 193, 7, 0.2);
            }
            
            .dataframe {
                border: 2px solid #77be5cff;
                border-collapse: collapse;
                width: 30%;
            }
            
            .dataframe th, .dataframe td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            
            .dataframe th {
                background-color: #77be5cff;
                color: white;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )

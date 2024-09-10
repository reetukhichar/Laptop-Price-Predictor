import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

col1, col2 = st.columns(2)

with col1:
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQLBLybOWE__N9886QHn4oa8ZDs91VVLldDKw&s')
with col2:
    st.title("Laptop Price Predictor")
# brand
company_list = df['Company'].unique().tolist()
company = st.selectbox('Brand', sorted(company_list))

# Type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input("Weight(in Kg)")

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size(in Inches)')

# resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1280x1024','1600x900','1680x1050','1280x800','1024x768','2560x1440','2560x1600','2304x1440','2560x1600','3840x2160','3200x1800','2880x1800'])

# cpu
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# ssd
ssd = st.selectbox('SSD(in GB)',[0, 8, 128, 256, 512, 1024])

# hdd
hdd = st.selectbox('HDD(in GB)',[0, 128, 256, 512, 1024, 2048])

# gpu
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS
os = st.selectbox('Operating System', df['OS'].unique())

if st.button('Predict Price'):
    # query
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])

    if screen_size < 10 or screen_size > 20:
        st.title("Enter the valid features")
    if weight < 1 or weight > 5:
        st.title("Enter the valid features")
    else:
        ppi = ((x_res**2) + (y_res**2))**0.5/screen_size

        query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,ssd,hdd,gpu,os])
        query = query.reshape(1,12)

        st.title('Predicted Price : ' + str(int(np.exp(pipe.predict(query)[0]))))
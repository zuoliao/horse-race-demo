import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import streamlit as st
import lightgbm as lgb
import glob


st.markdown('# horse-race-demo')

# Model
model_list = glob.glob('./models/*pkl')
model_list = [model.split('\\')[-1] for model in model_list]
selected_model = st.sidebar.selectbox('Model', model_list)
with open('./models/model_lightgbm_20240211.pkl', 'rb') as model_file:
  loaded_model = pickle.load(model_file)
  
# Input
input_list = glob.glob('./inputs/*.pkl')
index = st.sidebar.selectbox('No.', [n for n in range(len(input_list))])
with open(f'./inputs/test_input{index}.pkl', 'rb') as data:
    inputs = pickle.load(data)
# inputs = pd.DataFrame(inputs)
# st.table(inputs)

# Inference
outputs = loaded_model.predict(inputs)

fig, axes = plt.subplots(2,1,figsize=(12,8))
axes[0].plot(inputs[0])
axes[0].set_title('Input')

axes[1].bar([n for n in range(1, 19)], outputs[0])
axes[1].set_title('Predict')
axes[1].set_ylim([0.0, 1.0])
st.pyplot(fig)



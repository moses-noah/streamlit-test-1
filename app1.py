import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.figure_factory as ff
import plotly.express as px

def load_data():
    file = 'Unicorn_Companies.csv'
    df = pd.read_csv(file)
    df.loc[:, 'Valuation ($)'] = df.loc[:, 'Valuation'].str.replace('$', '').str.replace('B', '000000000').astype('int64')
    df.loc[:, 'Funding ($)'] = df.loc[:, 'Funding'].str.replace('Unknown', '-1').str.replace('$', '').str.replace('M', '000000').str.replace('B', '000000000').astype('int64')
    df.drop(columns=['Valuation', 'Funding'], axis=1, inplace=True)
    df['Date Joined'] = pd.to_datetime(df['Date Joined'])
    df.loc[:, 'Year Joined'] = df['Date Joined'].dt.year
    df.loc[:, 'Years to Unicorn status'] = df['Year Joined'] - df['Year Founded']
    df.loc[:, 'Count'] = 1

    return df

# load dataset
df = load_data()

st.title('Unicorn Companies App')
st.dataframe(df)

# create filters
# add a side bar and unique values from our columns
# industry filter
industry_list = df['Industry'].unique()
selected_industry = st.sidebar.multiselect('Industry', industry_list)
filtered_industry = df[df['Industry'].isin(selected_industry)]

# country filter
country_list = df['Country/Region'].unique()
selected_country = st.sidebar.multiselect('Country', country_list)
filtered_country = df[df['Country/Region'].isin(selected_country)]

# filter the data if industry or country is selected and if none
if selected_industry and selected_country:
    combined_table = df[df['Industry'].isin(selected_industry) & df['Country/Region'].isin(selected_country)]
    st.dataframe(combined_table)
elif selected_industry:
    st.dataframe(filtered_industry)
elif selected_country:
    st.dataframe(filtered_country)
else:
    st.dataframe(df)

# calculate metrics
no_of_companies = len(df)
total_valuation = f'$ {round(df['Valuation ($)'].sum()/1000000000,2)} B'
total_funding = df['Funding ($)'].sum()

# display the metrics using streamlit container / column components
col1, col2, col3 = st.columns(3)

with col1:
    st.metric('NO of companies', no_of_companies)
with col2:
    st.metric('Total Valuation', total_valuation)
with col3:
    st.metric('Total Funding', total_funding)

# create different chart
con = st.container()
with con:
    # matplotlib / seaborn
    st.subheader('Charts section')
    bar_plot_1 = sns.countplot(data = df, x = df['Industry'])
    plt.xticks(rotation= 45)
    plt.ylabel('No of Companies')
    st.pyplot(bar_plot_1.get_figure())

    # ploty line chart
    line_1 = px.bar(df, x = 'Year Joined', y = 'Count')
    st.plotly_chart(line_1)


# Assingment: create a side bar filter for 'City', 'Country/Region' and 'Continent'.
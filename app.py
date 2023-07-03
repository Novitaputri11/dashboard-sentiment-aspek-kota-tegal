import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Function untuk membuat barchart dengan filter kategori
###
def plot_sentiment_chart(dataset):
    """
    Fungsi untuk membuat chart menggunakan Plotly yang menampilkan jumlah sentimen 1 dan 0 pada setiap kategori.

    Args:
        dataset (pandas.DataFrame): DataFrame yang berisi dataset.

    """

    # Menghitung jumlah sentimen 1 dan 0 pada setiap kategori
    sentiment_counts = dataset.iloc[:, 2:].apply(pd.value_counts)

    # Membentuk DataFrame baru untuk chart
    sentiment_df = pd.DataFrame(sentiment_counts.stack()).reset_index()
    sentiment_df.columns = ['Kategori', 'Sentimen', 'Jumlah']

    # Menghapus kategori dengan nilai -1
    sentiment_df = sentiment_df[sentiment_df['Kategori'] != -1]

    # Mengubah label kategori
    sentiment_df['Kategori'] = sentiment_df['Kategori'].map({1: 'Positif', 0: 'Negatif'})

    # Membuat chart menggunakan Plotly
    fig = px.bar(sentiment_df, x='Kategori', y='Jumlah', color='Sentimen',
                 barmode='group', title='Jumlah Sentimen Positif dan Negatif pada Seluruh Kategori')
    fig.update_layout(xaxis_title='Kategori', yaxis_title='Jumlah')

    # Menampilkan chart menggunakan Streamlit
    st.plotly_chart(fig)
    
def plot_sentiment_chart_by_category(dataset, category):
    """
    Fungsi untuk membuat chart menggunakan Plotly yang menampilkan jumlah sentimen 1 dan 0 pada kategori yang spesifik.

    Args:
        dataset (pandas.DataFrame): DataFrame yang berisi dataset.
        category (str): Nama kolom kategori yang ingin ditampilkan.

    """

    # Menghitung jumlah sentimen 1 dan 0 pada kategori yang spesifik
    sentiment_counts = dataset[category].value_counts()

    # Membentuk DataFrame baru untuk chart
    sentiment_df = pd.DataFrame(sentiment_counts).reset_index()
    sentiment_df.columns = ['Sentimen', 'Jumlah']
    sentiment_df['Sentimen'] = sentiment_df['Sentimen'].map({1: 'Positif', 0: 'Negatif'})

    # Membuat chart menggunakan Plotly
    fig = px.bar(sentiment_df, x='Sentimen', y='Jumlah',
                 color='Sentimen', title=f'Jumlah Sentimen Positif dan Negatif pada Kategori {category}')
    fig.update_layout(xaxis_title='Sentimen', yaxis_title='Jumlah')

    # Menampilkan chart menggunakan Streamlit
    st.plotly_chart(fig)
    
def create_line_plot(data, category):
    # Filter data berdasarkan kategori dan negatif count != 0
    category_data = data[(data[category] != -1)]
    
    # Menghitung jumlah tweet positif dan negatif per tanggal
    positive_counts = category_data[category_data[category] == 1].groupby('tanggal').size()
    negative_counts = category_data[category_data[category] == 0].groupby('tanggal').size()
    
    # Menggabungkan kedua seri menjadi satu dataframe
    plot_data = pd.DataFrame({'Positif': positive_counts, 'Negatif': negative_counts}).reset_index()
    
    # Membuat line plot menggunakan Plotly
    fig = px.line(plot_data, x='tanggal', y=['Positif', 'Negatif'], title='Line Plot Positif dan Negatif berdasarkan Kategori: ' + category)
    
    # Mengubah warna label
    fig.update_traces(line=dict(color='blue'), selector=dict(name='Positif'))
    fig.update_traces(line=dict(color='red'), selector=dict(name='Negatif'))
    
    fig.update_layout(legend=dict(orientation="h",
                                  yanchor="top",  # Mengatur anchor ke atas
                                  y=1.0,         # Mengatur posisi y ke 1 (paling atas)
                                  xanchor="right",
                                  x=1))          # Mengatur posisi x ke 1 (paling kanan)

    # Menampilkan plot menggunakan Streamlit
    st.plotly_chart(fig)
    

def create_line_plot_with_date(data, category, start_date=None, end_date=None):
    data['tanggal']= pd.to_datetime(data['tanggal'])
    # Filter data berdasarkan kategori dan tanggal
    category_data = data[(data[category] != -1)]
    
    if start_date is not None:
        start_date = datetime.combine(start_date, datetime.min.time())
        category_data = category_data[category_data['tanggal'] >= start_date]
    if end_date is not None:
        end_date = datetime.combine(end_date, datetime.min.time())
        category_data = category_data[category_data['tanggal'] <= end_date]
    
    # Menghitung jumlah tweet positif dan negatif per tanggal
    positive_counts = category_data[category_data[category] == 1].groupby('tanggal').size()
    negative_counts = category_data[category_data[category] == 0].groupby('tanggal').size()
    
    # Menggabungkan kedua seri menjadi satu dataframe
    plot_data = pd.DataFrame({'Positif': positive_counts, 'Negatif': negative_counts}).reset_index()
    
    # Membuat line plot menggunakan Plotly
    fig = px.line(plot_data, x='tanggal', y=['Positif', 'Negatif'], title='Line Plot Positif dan Negatif berdasarkan Kategori: ' + category)
    
    # Menampilkan plot menggunakan Streamlit
    st.plotly_chart(fig)
    
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove usernames
    text = re.sub(r'@[^\s]+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert text to lowercase
    text = text.lower()

    # Remove stop words
    factory = StopWordRemoverFactory()
    stop_words = factory.create_stop_word_remover()
    text = stop_words.remove(text)

    # Remove specific phrase
    text = re.sub(r'tegal', '', text)
    text = re.sub(r'kota', '', text)

    return text

def create_wordcloud(data, category):
    # Filter dataset by category
    filtered_data = data[data[category] == 1]

    # Concatenate all tweets in the selected category
    text = ' '.join(filtered_data['tweet'])

    # Preprocess the text
    text = preprocess_text(text)

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - {}'.format(category.capitalize()))
    st.pyplot(plt)

def main():
    #judul dan Logo Kota Tegal
    col1, col2 = st.columns((2, 8))

    with col1:
        # st.header("")
        image = Image.open("Logo-Kota-Tegal.png")
        # resized_image = image.resize((150, 150))
        st.image(image, caption="", use_column_width=True)

    with col2:
        st.header("Sentimen Analisis Kota Tegal Pada Aspek Wisata Hiburan, Pendidikan, Fasilitas Publik, dan Kuliner")

    # Membaca data dari dataset Twitter Sentiment Analysis
    data = pd.read_csv("data-26 Juni 2023.csv", sep=",")

    # Membuat select box untuk memilih kategori
    categories = ['wisata_hiburan', 'pendidikan', 'fasilitas_layanan_publik', 'kuliner']
    category = st.sidebar.selectbox("Pilih kategori:", ['Semua Kategori']+categories)

    if category== 'Semua Kategori' :
        #Bar Plot
        plot_sentiment_chart(data)
        
    else:
        plot_sentiment_chart_by_category(data, category)
        tanpa_filter, filter = st.tabs(["Tanpa Filter", "Filter"])
        with tanpa_filter:
            create_line_plot(data, category)
            create_wordcloud(data, category)
        
        with filter:            
            col1, col2 = st.columns((2, 8))
            with col1:
                start_date = st.date_input("Pilih Tanggal Awal", value=None)
                end_date = st.date_input("Pilih Tanggal Akhir", value=None)
            with col2:            
                create_line_plot_with_date(data, category, start_date, end_date)   
main()


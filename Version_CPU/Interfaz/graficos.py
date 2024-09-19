import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
def crear_grafico_lineal(data, x, y, titulo, xlabel, ylabel, ylim=None, rotar_etiquetas=False):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x=x, y=y, marker='o', palette='viridis', ci=None)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim:
        plt.ylim(ylim)
    if rotar_etiquetas:
        plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()
    
def crear_grafico_barras(data, x, titulo, xlabel, ylabel, order=None):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x=x, palette='viridis', order=order)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt.gcf())
    plt.clf()  # Limpiar la figura para el siguiente gráfico

def crear_grafico_histograma(data, x, bins, titulo, xlabel, ylabel, color='blue'):
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x=x, bins=bins, kde=True, color=color)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt.gcf())
    plt.clf()  # Limpiar la figura para el siguiente gráfico   
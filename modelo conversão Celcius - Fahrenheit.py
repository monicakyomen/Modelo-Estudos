#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import matplotlib.pyplot as plt


# In[2]:


logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# In[3]:


celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float) #features de entrada

fahrenheit_a = np.array([-40,  14, 32, 46.4, 59, 71.6, 100.4],  dtype=float) #buget de saída


# In[4]:


modelo = tf.keras.Sequential()


# In[5]:


modelo.add(tf.keras.layers.Dense(10,activation= tf.nn.relu,input_shape=[1]))
modelo.add(tf.keras.layers.Dense(10,activation= tf.nn.relu,input_shape=[1])) 
modelo.add(tf.keras.layers.Dense(1)) #camada de saida

 #Isso cria uma camada densa (fully connected layer) com 10 neurônios (ou unidades).
#activation=tf.nn.relu: Define a função de ativação da camada como ReLU (Rectified Linear Unit), que é comumente usada em camadas ocultas para introduzir não-linearidade.
#input_shape=[1]: Define o formato dos dados de entrada para a camada. Neste caso, [1] indica que cada exemplo de entrada é um array unidimensional com uma feature.


# In[6]:


modelo.compile(loss='mean_squared_error',optimizer='Adam') 

#Este parâmetro especifica a função de perda (loss function) que será utilizada durante o treinamento do modelo. 
#Este parâmetro especifica o otimizador (optimizer) que será usado para ajustar os pesos da rede durante o treinamento, com o objetivo de minimizar a função de perda. No caso de optimizer='Adam', o Adam é um otimizador popular e eficaz baseado em gradientes estocásticos. 


# In[8]:


historico = modelo.fit(celsius_q,fahrenheit_a,batch_size=100,epochs=3000) 
#batch_size=100: Este parâmetro define o tamanho do lote (batch size) a ser utilizado durante o treinamento.
#O treinamento da rede neural é realizado em lotes de dados, e o tamanho do lote determina quantos exemplos de treinamento são utilizados de uma vez para atualizar os pesos do modelo.

#epochs=1050: Este parâmetro define o número de épocas (epochs) para treinar o modelo. Uma época representa uma passagem completa por todo o conjunto de dados de treinamento. 


# In[9]:


#Plote o histórico de perda do treinamento
plt.figure(figsize=(10,5))
plt.plot(historico.history['loss'])
plt.title('Histórico de Perda do Modelo')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend(['Treinamento'], loc='upper right')
plt.show()


# In[10]:


#Novos valores de temperatura em Celsius que você deseja testar
novos_valores_celsius = np.array([20, 25, 30], dtype=float)


# In[11]:


previsoes = modelo.predict(novos_valores_celsius)


# In[12]:


#Imprima as previsões
for i, valor_celsius in enumerate(novos_valores_celsius):
    print(f"{valor_celsius} graus Celsius é igual a {previsoes[i]} graus Fahrenheit de acordo com o modelo.")


# In[59]:


f_real = 68


# In[60]:


f_previsto = 67.774826


# In[61]:


erro_absoluto = abs(f_real-f_previsto)


# In[62]:


print("Error absoluto para 20C:", erro_absoluto, "graus f")


# In[ ]:





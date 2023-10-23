import streamlit as st
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import matplotlib.pyplot as plt

st.markdown(
    """
# Reporte de ventas de zapatos :shoe:

Este reporte muestra las ventas de zapatos durante un año.

En el reporte se identifica un crecimiento en las ventas de zapatos en el mes\
 de diciembre.

## Datos

A continuación cargamos algunos datos y los desplegamos.

Estos datos indican ventas de zapato por años en la empresa.
Es decir, para un vendedor que lleva 5 años en la empresa, se indica cuántos\
 zapatos vende en promedio por año.
""")

np.random.seed(1)
st.sidebar.title("Predicción de ventas :shoe:")
age = st.sidebar.slider("Años en la empresa", 0, 50, 20)

st.sidebar.title("Hiperparámetros :computer:")
epochs = st.sidebar.slider("Épocas de entrenamiento", 1000, 5000, 2000)

x = np.linspace(0, 50, 51)
y = x + 20 * np.random.random(len(x))

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))
fig.update_layout(
    title="Ventas de pares de zapatos/Años trabajados",
    xaxis_title="Años trabajados en la empresa",
    yaxis_title="Ventas de pares de zapatos"
)

st.plotly_chart(fig)

st.markdown(
    """
## Modelado de Datos

En los datos se observa un comportamiento lineal, por lo que se realiza un\
 ajuste de una línea recta con una regresión lineal simple.

$$y = m * x + b$$

""")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model.compile(optimizer=tf.keras.optimizers.Adam(0.1),  loss='mse')

history = model.fit(x, y, epochs=epochs, verbose=0)

st.write("Pérdida final:", history.history['loss'][-1])

x_loss = np.linspace(1, len(history.history['loss']),
                     len(history.history['loss']))
y_loss = history.history['loss']

fig = plt.figure()
plt.plot(x_loss, y_loss)
st.pyplot(fig)

y_pred = model.predict(x)
y_pred = y_pred.reshape(-1)

st.markdown(
    f"""
## Resultado

Podemos ver que después de crear nuestro modelo, y entrenar por {epochs} épocas
nuestra red de 1 neurona, el ajuste lineal sobre nuestros datos
es el siguiente:


""")
slope = (y_pred[1]-y_pred[0]/x[1]-x[0])
fig_2 = go.Figure()
fig_2.add_trace(go.Scatter(x=x, y=y, mode='markers', name="Datos"))
fig_2.add_trace(go.Scatter(x=x, y=y_pred, mode='lines',
                           name=f"Ajuste con pendiente {slope}"))
fig_2.update_layout(
    title="Ajuste Lineal",
    xaxis_title="Años trabajados en la empresa",
    yaxis_title="Ventas de pares de zapatos"
)
st.plotly_chart(fig_2)

st.write(f"Años en la empresa: {age}")
st.write(f"Se espera vender {round(model.predict([age])[0][0])} zapatos.")

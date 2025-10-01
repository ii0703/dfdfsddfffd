import pandas as pd
from prophet import Prophet
import sys

# Leer el archivo CSV de entrada
input_csv = sys.argv[1] if len(sys.argv) > 1 else 'ventas.csv'
df = pd.read_csv(input_csv)

# Renombrar columnas para Prophet
df.columns = ['ds', 'y']

# Crear y ajustar el modelo Prophet
model = Prophet()
model.fit(df)

# Crear dataframe futuro para pronóstico (por defecto 30 días)
futuro = model.make_future_dataframe(periods=30)
pronostico = model.predict(futuro)

# Unir datos reales y pronosticados

# Asegurar que las columnas 'ds' sean datetime en ambos dataframes
futuro['ds'] = pd.to_datetime(futuro['ds'])
pronostico['ds'] = pd.to_datetime(pronostico['ds'])
df['ds'] = pd.to_datetime(df['ds'])

resultado = pd.merge(futuro, pronostico[['ds', 'yhat']], on='ds', how='left')
resultado = pd.merge(resultado, df, on='ds', how='left')
resultado = resultado.rename(columns={'ds': 'fecha', 'y': 'ventas', 'yhat': 'ventas_pronosticadas'})

# Guardar en Excel
resultado[['fecha', 'ventas', 'ventas_pronosticadas']].to_excel('forecast_resultado.xlsx', index=False)

print('Pronóstico generado en forecast_resultado.xlsx')

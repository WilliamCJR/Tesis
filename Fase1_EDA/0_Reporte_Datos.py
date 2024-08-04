import pandas as pd
from ydata_profiling import ProfileReport

data = pd.read_csv('DatosFiltradosDisponibilidadCampos.csv', delimiter=';')
profile = ProfileReport(data, title='Reporte de Datos', explorative=True)
profile.to_file("reporte_datos.html")
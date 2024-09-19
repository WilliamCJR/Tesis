def retorna_info():
    dt_info = {
    "CAMPO": [
        "ANNO", "MES", "ID_EMPRESA", "DESC_EMPRESA", "ID_ELEMENTO", 
        "ELEMENTO_PACTADO", "ID_TECNOLOGIA", "DESC_TECNOLOGIA", 
        "ID_DEPARTAMENTO", "DESC_DEPARTAMENTO", "ID_MUNICIPIO", 
        "DESC_MUNICIPIO", "ID_DIVISION_ADMINISTRATIVA", 
        "DESC_DIVISION_ADMINISTRATIVA", "ID_ZONA", 
        "TIEMPO_TOTAL_INDISPONIBILIDAD", "PORCENTAJE_DE_DISPONIBILIDAD", 
        "TRANSMISION_SATELITAL"
    ],
    "TIPO DE DATO": [
        "ENTERO", "ENTERO", "ENTERO", "TEXTO", "ENTERO", 
        "BOOLEANO", "ENTERO", "TEXTO", 
        "ENTERO", "TEXTO", "ENTERO", 
        "TEXTO", "ENTERO", 
        "TEXTO", "ENTERO", 
        "FLOTANTE", "FLOTANTE", 
        "BOOLEANO"
    ],
    "DESCRIPCIÓN": [
        "Corresponde al año para el cual se reporta la información. Campo numérico entero, serie de cuatro dígitos.",
        "Corresponde al mes del año en el que se realizó el cálculo del indicador.",
        "Nit de la empresa que está reportando.",
        "Corresponde a la razón social del PRST que está reportando.",
        "Código único generado por la CRC que identifica el elemento de red de acceso fija o móvil.",
        "Indica si el elemento de red hace parte del cumplimiento de acuerdos del nivel de servicio dentro del contrato donde se haya negociado la totalidad de las condiciones de la prestación del servicio (S/N).",
        "Código del tipo de tecnología móviles: 1. 2G. 2. 3G. 3. 4G. Tipo de tecnología fijas cableadas: 4. CMTS (para redes con tecnología HFC). 5. OLT (para redes con tecnología PON).",
        "Nombre del tipo de tecnología móviles: 1. 2G. 2. 3G. 3. 4G. Tipo de tecnología fijas cableadas: 4. CMTS (para redes con tecnología HFC). 5. OLT (para redes con tecnología PON).",
        "Corresponde a la codificación DIVIPOLA para el departamento presente en el sistema de consulta del DANE.",
        "Corresponde al nombre del departamento al cual pertenece el municipio donde se realiza la medición del indicador.",
        "Ubicación geográfica de la estación base o del equipo terminal de acceso para red fija. Se tiene en cuenta los 32 departamentos y la ciudad de Bogotá D.C. Los municipios se identifican de acuerdo con la división político-administrativa de Colombia, DIVIPOLA, presente en el sistema de consulta del DANE. Para aquellas capitales con una población mayor a 500.000 habitantes se debe relacionar las divisiones administrativas, esto es localidades, o comunas, de acuerdo con el ordenamiento territorial de cada una.",
        "Corresponde al nombre del municipio donde se realiza la medición del indicador.",
        "Corresponde a la Codificación del DANE para las divisiones administrativas. Para registros que no requieren especificar división administrativa se visualiza 'N/A'.",
        "Corresponde a las divisiones administrativas de las capitales de departamento que posean una población mayor a quinientos mil (500.000) habitantes, según lo establecido en el DANE. De lo contrario se visualiza 'N/A'.",
        "Para efectos de la diferenciación por zonas, se deberán tomar las definiciones encontradas en el TÍTULO I: 101. Zona 1, 102. Zona 2.",
        "Es el tiempo total en minutos en que el elemento de red estuvo fuera de servicio, o no se encontró disponible.",
        "Es igual al 100% menos la relación porcentual entre la cantidad de minutos en los que el elemento de red no estuvo disponible en el mes de reporte, y la cantidad total de minutos del periodo de reporte.",
        "Indica 'S' en los casos en que la estación base tiene transmisión satelital,de lo contrario indica 'N'. - (S/N)."
    ]
    } 
    return dt_info
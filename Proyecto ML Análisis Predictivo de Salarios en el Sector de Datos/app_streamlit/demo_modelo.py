import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Carga de Datos y Modelo

@st.cache_data
def load_data():
    """Carga el dataframe procesado para obtener las opciones de los selectores."""
    df_final = pd.read_csv('salarios_codificado.csv')
    return df_final

@st.cache_resource
def load_model():
    """Carga el modelo de machine learning pre-entrenado."""
    with open('mi_modelo_salarios.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Cargar los datos y el modelo al iniciar la app
df_final = load_data()
model = load_model()

# Extraer listas de columnas para los selectores
paises_original = [col.replace('loc_', '') for col in df_final.columns if 'loc_' in col]
industrias_original = [col.replace('industry_', '') for col in df_final.columns if 'industry_' in col]
tipos_empleo = [col.replace('employment_type_', '') for col in df_final.columns if 'employment_type_' in col]
puestos = [col.replace('job_title_', '') for col in df_final.columns if 'job_title_' in col]
skills = [col.replace('skill_', '').replace('_', ' ') for col in df_final.columns if 'skill_' in col]

# DICCIONARIOS DE TRADUCCIÓN
# Mapeos para mostrar etiquetas amigables en español
exp_map_es = {0: 'Junior (EN)', 1: 'Intermedio (MI)', 2: 'Senior (SE)', 3: 'Ejecutivo (EX)'}
edu_map_es = {0: 'Técnico / Asociado', 1: 'Grado / Licenciatura', 2: 'Máster', 3: 'Doctorado (PhD)'}
size_map_es = {0: 'Pequeña (S)', 1: 'Mediana (M)', 2: 'Grande (L)'}

# Mapeo para países
country_map = {
    'Austria': 'Austria', 'Finland': 'Finlandia', 'France': 'Francia',
    'Germany': 'Alemania', 'Ireland': 'Irlanda', 'Netherlands': 'Países Bajos'
}
# Mapeo inverso para obtener el valor original a partir de la selección en español
country_map_rev = {v: k for k, v in country_map.items()}

# Mapeo para industrias
industry_map = {
    'Consulting': 'Consultoría', 'Education': 'Educación', 'Energy': 'Energía',
    'Finance': 'Finanzas', 'Gaming': 'Videojuegos', 'Government': 'Gobierno',
    'Healthcare': 'Salud', 'Manufacturing': 'Manufactura', 'Media': 'Medios',
    'Real Estate': 'Inmobiliaria', 'Retail': 'Retail', 'Technology': 'Tecnología',
    'Telecommunications': 'Telecomunicaciones', 'Transportation': 'Transporte'
}
industry_map_rev = {v: k for k, v in industry_map.items()}


# Interfaz de Usuario

st.title("Demo del Modelo Predictivo de Salarios")
st.markdown("Introduce los detalles de una oferta de empleo para obtener una estimación salarial en el mercado europeo.")

# Barra Lateral para Inputs del Usuario
st.sidebar.header("Parámetros del Puesto")

# Selectores con opciones en español
exp_level = st.sidebar.selectbox('Nivel de Experiencia', options=list(exp_map_es.keys()), format_func=lambda x: exp_map_es[x], index=2)
years_exp = st.sidebar.slider('Años de Experiencia', 0, 20, 5)
edu_level = st.sidebar.selectbox('Nivel Educativo Requerido', options=list(edu_map_es.keys()), format_func=lambda x: edu_map_es[x], index=1)

st.sidebar.divider()

comp_size = st.sidebar.selectbox('Tamaño de la Empresa', options=list(size_map_es.keys()), format_func=lambda x: size_map_es[x], index=1)
# El usuario ve español, pero guardamos la clave en inglés para el modelo
comp_loc_es = st.sidebar.selectbox('País de la Empresa', options=list(country_map.values()), index=3) # Alemania por defecto
comp_loc = country_map_rev[comp_loc_es]

industry_es = st.sidebar.selectbox('Industria', options=list(industry_map.values()), index=11) # Tecnología por defecto
industry = industry_map_rev[industry_es]

st.sidebar.divider()

job_title = st.sidebar.selectbox('Puesto de Trabajo (Top 20)', puestos, index=puestos.index("Data Scientist"))
emp_type = st.sidebar.selectbox('Tipo de Contrato', tipos_empleo, index=tipos_empleo.index("FT"))
remote_ratio = st.sidebar.slider('Ratio de Teletrabajo', 0.0, 1.0, 0.5, 0.5)
benefits_score = st.sidebar.slider('Puntuación de Beneficios', 5.0, 10.0, 7.5)
is_international = st.sidebar.checkbox('Contrato Internacional (Residencia != Ubicación Empresa)')

st.sidebar.divider()

selected_skills = st.sidebar.multiselect('Habilidades Requeridas (Top 20)', skills, default=['Python', 'SQL', 'TensorFlow'])

# Lógica de Predicción
# Crear un diccionario con todas las columnas y valores por defecto
input_data = {col: 0 for col in df_final.drop(columns=['salary_EUR']).columns}

# Actualizar con los valores del usuario
input_data['experience_level_encoded'] = exp_level
input_data['years_experience'] = years_exp
input_data['education_encoded'] = edu_level
input_data['company_size'] = comp_size
input_data['remote_ratio'] = remote_ratio
input_data['benefits_score'] = benefits_score
input_data['is_international'] = int(is_international)

# Actualizar columnas One-Hot 
if f'loc_{comp_loc}' in input_data:
    input_data[f'loc_{comp_loc}'] = 1
if f'industry_{industry}' in input_data:
    input_data[f'industry_{industry}'] = 1
if f'employment_type_{emp_type}' in input_data:
    input_data[f'employment_type_{emp_type}'] = 1
if f'job_title_{job_title}' in input_data:
    input_data[f'job_title_{job_title}'] = 1

# Actualizar skills
for skill in selected_skills:
    skill_col = f"skill_{skill.replace(' ', '_')}"
    if skill_col in input_data:
        input_data[skill_col] = 1

# Convertir a DataFrame en el orden correcto
input_df = pd.DataFrame([input_data])
input_df = input_df[df_final.drop(columns=['salary_EUR']).columns] # Asegurar orden

# Realizar la predicción automáticamente al cambiar un widget
prediction = model.predict(input_df)[0]

# Mostrar el Resultado Principal
st.header("Resultado de la Predicción")
st.metric(label="Salario Anual Bruto Estimado en Europa", value=f"€ {prediction:,.2f}")


with st.expander("Ver los datos de entrada utilizados para la predicción"):
    # Lista de etiquetas para las características seleccionadas
    labels = [
        "Nivel de Experiencia", "Años de Experiencia", "Nivel Educativo",
        "Tamaño de Empresa", "País", "Industria",
        "Puesto de Trabajo", "Tipo de Contrato", "Ratio de Teletrabajo",
        "Puntuación de Beneficios", "Contrato Internacional", "Habilidades"
    ]
    
    # Lista de los valores seleccionados por el usuario
    values = [
        exp_map_es[exp_level], years_exp, edu_map_es[edu_level],
        size_map_es[comp_size], comp_loc_es, industry_es,
        job_title, emp_type, f"{remote_ratio*100:.0f}%", benefits_score,
        "Sí" if is_international else "No",
        ", ".join(selected_skills) if selected_skills else "Ninguna"
    ]

    # Crear el diccionario y el DataFrame
    display_data = {
        'Característica': labels,
        'Valor Seleccionado': values
    }
    
    display_df = pd.DataFrame(display_data)
    
    # Mostrar la tabla de resumen
    st.table(display_df.set_index("Característica"))
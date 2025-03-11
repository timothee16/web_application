import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
from datetime import datetime


@st.cache_resource
def get_db_connection():
    """connection to SQLite database"""
    current_dir = Path(__file__).parent.absolute()
    db_path = current_dir.parent / 'database' / 'mimic.db'
    return sqlite3.connect(str(db_path), check_same_thread=False)

@st.cache_data(show_spinner=False)
def get_tables_with_subject_id(_conn):
    """tables that have a subject_id column"""
    with st.spinner('Analyzing database structure...'):
        cursor = _conn.cursor()
        cursor.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' 
            AND name NOT LIKE 'sqlite_%'
        """)
        tables = [table[0] for table in cursor.fetchall()]
        
        
        tables_with_subject_id = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info([{table}])")
            columns = [info[1] for info in cursor.fetchall()]
            if "subject_id" in columns:
                tables_with_subject_id.append(table)
        
        return tables_with_subject_id

@st.cache_data(show_spinner=False)
def get_table_columns(_conn, table):
    """get columns of a table"""
    cursor = _conn.cursor()
    cursor.execute(f"PRAGMA table_info([{table}])")
    return [info[1] for info in cursor.fetchall()]

@st.cache_data(show_spinner=False)
def get_patient_ids(_conn, table):
    """get unique patient IDs (here it's limited to 1000 for test)"""
    with st.spinner('Loading patient IDs...'):
        query = f"""
        SELECT DISTINCT subject_id
        FROM [{table}]
        ORDER BY subject_id
        LIMIT 1000
        """
        return pd.read_sql_query(query, _conn)["subject_id"].tolist()

@st.cache_data(show_spinner=False)
def get_admission_ids(_conn, subject_id):
    """get admission IDs for a given patient"""
    with st.spinner('Loading admissions...'):
        query = """
        SELECT DISTINCT hadm_id, admittime, dischtime, admission_type, admission_location
        FROM admissions
        WHERE subject_id = ?
        ORDER BY admittime DESC
        """
        df = pd.read_sql_query(query, _conn, params=[subject_id])
        if not df.empty:
            df['admittime'] = pd.to_datetime(df['admittime'])
            df['dischtime'] = pd.to_datetime(df['dischtime'])
            df['duration'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / 86400
            df['label'] = df.apply(lambda x: f"{x['hadm_id']} - {x['admittime'].strftime('%d/%m/%Y')} ({x['admission_type']})", axis=1)
            return df
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_patient_data_by_admission(_conn, table, subject_id, hadm_id):
    """Get patient data for a specific admission"""
    try:
        cursor = _conn.cursor()
        cursor.execute(f"PRAGMA table_info([{table}])")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "hadm_id" in columns:
            query = f"""
            SELECT *
            FROM [{table}]
            WHERE subject_id = ? AND hadm_id = ?
            """
            return pd.read_sql_query(query, _conn, params=[subject_id, hadm_id])
        else:
            query = f"""
            SELECT *
            FROM [{table}]
            WHERE subject_id = ?
            """
            return pd.read_sql_query(query, _conn, params=[subject_id])
    except Exception as e:
        st.error(f"Error querying table {table}: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_patient_demographics(_conn, subject_id):
    """get demographic information for a patient"""
    query = """
    SELECT p.subject_id, p.gender, p.anchor_age as age, 
           p.anchor_year, p.dod,
           a.insurance, a.language, a.marital_status, a.race
    FROM patients p
    LEFT JOIN admissions a ON p.subject_id = a.subject_id
    WHERE p.subject_id = ?
    LIMIT 1
    """
    try:
        return pd.read_sql_query(query, _conn, params=[subject_id])
    except Exception as e:
        st.error(f"Error retrieving demographic data: {e}")
        cursor = _conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patients'")
        if not cursor.fetchone():
            st.warning("The 'patients' table does not exist in the database")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_medication_data(_conn, subject_id, hadm_id):
    """get medication data with enriched information"""

    try:
        query = """
        SELECT e.charttime, e.medication, e.scheduletime, e.dose_given, e.dose_unit, e.route
        FROM emar e
        WHERE e.subject_id = ? AND e.hadm_id = ?
        ORDER BY e.charttime
        """
        df = pd.read_sql_query(query, _conn, params=[subject_id, hadm_id])
        
        if not df.empty:
            df['charttime'] = pd.to_datetime(df['charttime'])
            df['scheduletime'] = pd.to_datetime(df['scheduletime'])
            return df
    except Exception:
        pass
    
    try:
        query = """
        SELECT p.startdate as charttime, p.drug as medication, 
               p.enddate as scheduletime, p.dose_val_rx as dose_given,
               p.dose_unit_rx as dose_unit, p.route
        FROM prescriptions p
        WHERE p.subject_id = ? AND p.hadm_id = ?
        ORDER BY p.startdate
        """
        df = pd.read_sql_query(query, _conn, params=[subject_id, hadm_id])
        
        if not df.empty:
            df['charttime'] = pd.to_datetime(df['charttime'])
            df['scheduletime'] = pd.to_datetime(df['scheduletime'])
            return df
    except Exception:
        pass
    
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_time_series_data(_conn, subject_id, hadm_id):
    """get time series data for a patient and admission"""
    time_series_data = {}
    
    # labevents
    try:
        query = """
        SELECT l.charttime, d.label, l.valuenum
        FROM labevents l
        JOIN d_labitems d ON l.itemid = d.itemid
        WHERE l.subject_id = ? AND l.hadm_id = ?
        AND l.valuenum IS NOT NULL
        ORDER BY l.charttime
        """
        df = pd.read_sql_query(query, _conn, params=[subject_id, hadm_id])
        if not df.empty:
            df['charttime'] = pd.to_datetime(df['charttime'])
            time_series_data['labevents'] = df
    except Exception as e:
        st.error(f"Error retrieving lab data: {e}")
    
    # vital signs
    try:
        query = """
        SELECT c.charttime, d.label, c.valuenum
        FROM chartevents c
        JOIN d_items d ON c.itemid = d.itemid
        WHERE c.subject_id = ? AND c.hadm_id = ?
        AND c.valuenum IS NOT NULL
        AND d.category IN ('Vitals', 'Labs', 'Routine Vital Signs')
        ORDER BY c.charttime
        """
        df = pd.read_sql_query(query, _conn, params=[subject_id, hadm_id])
        if not df.empty:
            df['charttime'] = pd.to_datetime(df['charttime'])
            time_series_data['chartevents'] = df
    except Exception:
        pass

    # medications (emar)
    try:
        query = """
        SELECT charttime, medication, scheduletime
        FROM emar
        WHERE subject_id = ? AND hadm_id = ?
        ORDER BY charttime
        """
        df = pd.read_sql_query(query, _conn, params=[subject_id, hadm_id])
        if not df.empty:
            df['charttime'] = pd.to_datetime(df['charttime'])
            df['scheduletime'] = pd.to_datetime(df['scheduletime'])
            time_series_data['medications'] = df
    except Exception:
        pass
    
    return time_series_data

@st.cache_data(show_spinner=False)
def get_relevant_vital_signs(_conn, subject_id, hadm_id):
    """get the most relevant vital signs for a patient"""
    important_vitals_query = """
    SELECT c.charttime, d.label, c.valuenum
    FROM chartevents c
    JOIN d_items d ON c.itemid = d.itemid
    WHERE c.subject_id = ? AND c.hadm_id = ?
    AND c.valuenum IS NOT NULL
    AND d.label IN (
        'Heart Rate', 'Respiratory Rate', 'O2 saturation pulseoxymetry',
        'Non Invasive Blood Pressure systolic', 'Non Invasive Blood Pressure diastolic',
        'Temperature Fahrenheit', 'Temperature Celsius', 'Blood Pressure systolic',
        'Blood Pressure diastolic', 'Mean Arterial Pressure', 'Glasgow Coma Scale Total'
    )
    ORDER BY c.charttime
    """
    
    try:
        df = pd.read_sql_query(important_vitals_query, _conn, params=[subject_id, hadm_id])
        if not df.empty:
            df['charttime'] = pd.to_datetime(df['charttime'])
            
            # standardize vital sign names
            name_mapping = {
                'O2 saturation pulseoxymetry': 'Oxygen Saturation',
                'Non Invasive Blood Pressure systolic': 'Blood Pressure (Systolic)',
                'Non Invasive Blood Pressure diastolic': 'Blood Pressure (Diastolic)',
                'Blood Pressure systolic': 'Blood Pressure (Systolic)',
                'Blood Pressure diastolic': 'Blood Pressure (Diastolic)',
                'Temperature Fahrenheit': 'Temperature',
                'Temperature Celsius': 'Temperature',
                'Mean Arterial Pressure': 'MAP',
                'Glasgow Coma Scale Total': 'GCS'
            }
            
            df['label'] = df['label'].replace(name_mapping)
            
            # convert fahrenheit to celsius for standardization
            mask = df['label'] == 'Temperature'
            temp_f_mask = mask & df['valuenum'].between(95, 108)
            
            if temp_f_mask.any():
                df.loc[temp_f_mask, 'valuenum'] = (df.loc[temp_f_mask, 'valuenum'] - 32) * 5/9
            
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving vital signs: {e}")
        return pd.DataFrame()


def create_time_series_chart(data, title, y_label, start_date=None, end_date=None):
    """creates a time series chart with proper date range"""
    if data.empty:
        return None
    
    # top 10 items
    top_items = data['label'].value_counts().nlargest(10).index.tolist()
    filtered_data = data[data['label'].isin(top_items)]
    
    min_date = start_date if start_date else filtered_data['charttime'].min()
    max_date = end_date if end_date else filtered_data['charttime'].max()
    
    fig = px.line(
        filtered_data, 
        x='charttime', 
        y='valuenum', 
        color='label',
        title=title,
        labels={'charttime': 'Date/Time', 'valuenum': y_label, 'label': 'Measurement'}
    )
    
    fig.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=80, b=120),
        xaxis_range=[min_date, max_date],
        xaxis_title="Date/Time",
        yaxis_title=y_label,
        plot_bgcolor='rgba(248,249,250,0.5)',
        title_font=dict(size=18)
    )
    
    return fig

def create_medication_timeline(data, start_date=None, end_date=None):
    """medication timeline chart"""
    if data.empty:
        return None
    
    # group data by date and medication
    data['date'] = data['charttime'].dt.floor('D')
    grouped = data.groupby(['date', 'medication']).size().reset_index(name='count')
    
    min_date = start_date if start_date else grouped['date'].min()
    max_date = end_date if end_date else grouped['date'].max()
    
    fig = px.line(
        grouped, 
        x='date', 
        y='count', 
        color='medication',
        title="Medications Administered Over Time",
        labels={'date': 'Date', 'count': 'Number of Administrations', 'medication': 'Medication'}
    )
    
    fig.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=80, b=100),
        xaxis_range=[min_date, max_date],
        plot_bgcolor='rgba(248,249,250,0.5)',
        title_font=dict(size=18)
    )
    
    return fig
    

def create_combined_medication_vital_chart(med_data, vital_data, medication_name, vital_sign_name, start_date=None, end_date=None):
    """chart combining medication administration and vital sign"""
    if med_data.empty or vital_data.empty:
        return None
    
    if medication_name not in med_data['medication'].unique():
        return None
    
    if vital_sign_name not in vital_data['label'].unique():
        return None
    
    # filter data for specific medication and vital sign
    med_filtered = med_data[med_data['medication'] == medication_name]
    vital_filtered = vital_data[vital_data['label'] == vital_sign_name]
    
    if med_filtered.empty or vital_filtered.empty:
        return None
    
    min_date = start_date if start_date else min(med_filtered['charttime'].min(), vital_filtered['charttime'].min())
    max_date = end_date if end_date else max(med_filtered['charttime'].max(), vital_filtered['charttime'].max())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=vital_filtered['charttime'],
        y=vital_filtered['valuenum'],
        mode='lines+markers',
        name=vital_sign_name,
        line=dict(color='blue')
    ))
    
    for i, row in med_filtered.iterrows():
        fig.add_shape(
            type="line",
            x0=row['charttime'], 
            y0=vital_filtered['valuenum'].min(), 
            x1=row['charttime'], 
            y1=vital_filtered['valuenum'].max(),
            line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dash"),
        )
    
    fig.add_trace(go.Scatter(
        x=med_filtered['charttime'],
        y=[vital_filtered['valuenum'].min() + (vital_filtered['valuenum'].max() - vital_filtered['valuenum'].min()) * 0.05] * len(med_filtered),
        mode='markers',
        marker=dict(
            symbol='triangle-up',
            size=12,
            color='red',
        ),
        name=f'{medication_name} Administration',
        hoverinfo='x+text',
        hovertext=[f"{medication_name} administered at {t.strftime('%Y-%m-%d %H:%M')}" for t in med_filtered['charttime']]
    ))
    
    fig.update_layout(
        title=f"{vital_sign_name} and {medication_name} Administration Over Time",
        xaxis_title="Date/Time",
        yaxis_title=vital_sign_name,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=80),
        xaxis_range=[min_date, max_date],
        plot_bgcolor='rgba(248,249,250,0.5)',
        title_font=dict(size=18)
    )
    
    return fig


def format_demographic_data(demographics_df):
    """format demographic data for display"""
    if demographics_df.empty:
        return "No demographic data available"
    
    demo = demographics_df.iloc[0]
    
    def format_long_value(value, max_length=40):
        if isinstance(value, str) and len(value) > max_length:
            return f"{value[:max_length]}..."
        return value if pd.notna(value) else "N/A"
    
    info = {
        "**Age**": f"{demo.get('age', 'N/A')} years",
        "**Gender**": format_long_value(demo.get('gender')),
        "**Race**": format_long_value(demo.get('race')),
        "**Marital Status**": format_long_value(demo.get('marital_status')),
        "**Language**": format_long_value(demo.get('language')),
        "**Insurance**": format_long_value(demo.get('insurance'))
    }
    
    return info

def display_medication_dashboard(medication_data, vital_signs_data, admit_time, discharge_time):
    """display a complete dashboard for medications"""
    if medication_data.empty:
        st.info("No medication data available for this patient")
        return
    
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.subheader("Medication Dashboard")
    
    st.markdown("### Medication Overview")
    med_counts = medication_data['medication'].value_counts().reset_index()
    med_counts.columns = ['Medication', 'Count']
    
    fig = px.bar(
        med_counts.head(10),
        x='Count',
        y='Medication',
        orientation='h',
        title='Top 10 Medications by Frequency',
        labels={'Count': 'Number of Administrations', 'Medication': ''},
        color='Count',
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=80, b=40),
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(248,249,250,0.5)',
        title_font=dict(size=18)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Medication Timeline")
    med_timeline = create_medication_timeline(medication_data, admit_time, discharge_time)
    if med_timeline:
        st.plotly_chart(med_timeline, use_container_width=True)
    
    if not vital_signs_data.empty and not medication_data.empty:
        st.markdown("### Medication-Vital Sign Analysis")
        
        available_meds = medication_data['medication'].unique().tolist()
        
        if not available_meds:
            st.warning("No medications available for analysis.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                top_meds = medication_data['medication'].value_counts().nlargest(5).index.tolist()
                if top_meds:
                    selected_med = st.selectbox("Select Medication:", top_meds, key="med_dash_med_select")
                else:
                    st.warning("No medications found in the data.")
                    selected_med = None
            
            with col2:
                available_vitals = vital_signs_data['label'].unique().tolist()
                if available_vitals:
                    selected_vital = st.selectbox("Select Vital Sign:", available_vitals, key="med_dash_vital_select")
                else:
                    st.warning("No vital signs found in the data.")
                    selected_vital = None
            
            if selected_med and selected_vital:
                combined_chart = create_combined_medication_vital_chart(
                    medication_data,
                    vital_signs_data,
                    selected_med,
                    selected_vital,
                    admit_time,
                    discharge_time
                )
                
                if combined_chart:
                    st.plotly_chart(combined_chart, use_container_width=True)
                    st.info("This chart shows the relationship between medication administration (red triangles/vertical lines) and the selected vital sign.")
                else:
                    st.warning("Insufficient data to create the combined viz.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_patient_summary(conn, subject_id, hadm_id):
    """display summary of patient information"""
    demographics = get_patient_demographics(conn, subject_id)
    
    # admission
    admission_query = """
    SELECT admission_type, admission_location, discharge_location, 
           admittime, dischtime, insurance, language, marital_status
    FROM admissions
    WHERE subject_id = ? AND hadm_id = ?
    """
    admission = pd.read_sql_query(admission_query, conn, params=[subject_id, hadm_id])
    
    # diagnoses
    diagnostics_query = """
    SELECT d.icd_code, i.long_title
    FROM diagnoses_icd d
    JOIN d_icd_diagnoses i ON d.icd_code = i.icd_code AND d.icd_version = i.icd_version
    WHERE d.subject_id = ? AND d.hadm_id = ?
    ORDER BY d.seq_num
    """
    try:
        diagnostics = pd.read_sql_query(diagnostics_query, conn, params=[subject_id, hadm_id])
    except:
        diagnostics = pd.DataFrame()
    
    # procedures
    procedures_query = """
    SELECT p.icd_code, i.long_title
    FROM procedures_icd p
    JOIN d_icd_procedures i ON p.icd_code = i.icd_code AND p.icd_version = i.icd_version
    WHERE p.subject_id = ? AND p.hadm_id = ?
    ORDER BY p.seq_num
    """
    try:
        procedures = pd.read_sql_query(procedures_query, conn, params=[subject_id, hadm_id])
    except:
        procedures = pd.DataFrame()
    
    # display summary
    st.markdown('<div class="header-style">', unsafe_allow_html=True)
    
    # demographic info
    demo_info = format_demographic_data(demographics)
    
    col1, col2, col3 = st.columns(3)
    
    if isinstance(demo_info, dict):
        with col1:
            st.subheader("Patient Summary")
            for key, value in list(demo_info.items())[:2]:
                st.metric(key, value)
        
        with col2:
            st.subheader(" ")
            for key, value in list(demo_info.items())[2:4]:
                st.metric(key, value)
        
        with col3:
            st.subheader(" ")
            for key, value in list(demo_info.items())[4:]:
                st.metric(key, value)
    else:
        st.warning(demo_info)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # admission info
    if not admission.empty:
        adm = admission.iloc[0]
        
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.subheader("Admission Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            admit_date = pd.to_datetime(adm['admittime'])
            discharge_date = pd.to_datetime(adm['dischtime'])
            duration = (discharge_date - admit_date).days
            
            st.metric("**Admission Type**", adm['admission_type'])
            st.metric("**Length of Stay**", f"{duration} days")
            st.metric("**Source**", adm['admission_location'])
        
        with col2:
            st.metric("**Admission Date**", admit_date.strftime('%d/%m/%Y'))
            st.metric("**Discharge Date**", discharge_date.strftime('%d/%m/%Y'))
            st.metric("**Destination**", adm['discharge_location'])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # diagnoses
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.subheader("Diagnoses")
    
    if not diagnostics.empty:
        st.dataframe(
            diagnostics,
            column_config={
                "icd_code": "ICD Code",
                "long_title": "Description"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No diagnostic data available for this admission")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # procedures
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.subheader("Procedures")
    
    if not procedures.empty:
        st.dataframe(
            procedures,
            column_config={
                "icd_code": "ICD Code",
                "long_title": "Description"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No procedure data available for this admission")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_time_series_data(time_series_data, admit_time, discharge_time):
    """affiche les données de séries temporelles pour un patient"""
    if not time_series_data:
        st.info("No time series data available for this patient")
        return
    
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.subheader("Temporal Monitoring")
    
    if isinstance(admit_time, str):
        admit_time = pd.to_datetime(admit_time)
    if isinstance(discharge_time, str):
        discharge_time = pd.to_datetime(discharge_time)
    
    # date range slider for all charts
    date_range = [admit_time, discharge_time]
    
    # date range selector
    dates = st.slider(
        "Select Date Range for Charts",
        min_value=admit_time.to_pydatetime(),
        max_value=discharge_time.to_pydatetime(),
        value=(admit_time.to_pydatetime(), discharge_time.to_pydatetime()),
        format="DD/MM/YY HH:mm"
    )
    selected_start_date, selected_end_date = dates
    
    # lab results
    if 'labevents' in time_series_data and not time_series_data['labevents'].empty:
        lab_chart = create_time_series_chart(
            time_series_data['labevents'],
            "Lab Results Over Time",
            "Value",
            selected_start_date,
            selected_end_date
        )
        if lab_chart:
            st.plotly_chart(lab_chart, use_container_width=True)
    
    # vital signs
    if 'chartevents' in time_series_data and not time_series_data['chartevents'].empty:
        vitals_chart = create_time_series_chart(
            time_series_data['chartevents'],
            "Vital Signs Over Time",
            "Value",
            selected_start_date,
            selected_end_date
        )
        if vitals_chart:
            st.plotly_chart(vitals_chart, use_container_width=True)
    
    # medications
    if 'medications' in time_series_data and not time_series_data['medications'].empty:
        med_chart = create_medication_timeline(
            time_series_data['medications'],
            selected_start_date,
            selected_end_date
        )
        if med_chart:
            st.plotly_chart(med_chart, use_container_width=True)
        
        st.markdown("### Medication and Vital Sign Correlation")
        
        top_meds = time_series_data['medications']['medication'].value_counts().nlargest(5).index.tolist()
        
        vital_data = None
        if 'chartevents' in time_series_data and not time_series_data['chartevents'].empty:
            vital_data = time_series_data['chartevents']
        
        if not top_meds or vital_data is None or vital_data.empty:
            st.warning("Insufficient data for medication-vital sign correlation analysis.")
        else:
            # available vital signs
            available_vitals = vital_data['label'].unique().tolist()
            
            if not available_vitals:
                st.warning("No vital sign data available for correlation analysis.")
            else:
                # selectors for medication and vital sign
                col1, col2 = st.columns(2)
                with col1:
                    selected_med = st.selectbox("Select Medication:", top_meds, key="corr_med_select")
                with col2:
                    selected_vital = st.selectbox("Select Vital Sign:", available_vitals, key="corr_vital_select")
                
                combined_chart = create_combined_medication_vital_chart(
                    time_series_data['medications'],
                    vital_data,
                    selected_med,
                    selected_vital,
                    selected_start_date,
                    selected_end_date
                )
                
                if combined_chart:
                    st.plotly_chart(combined_chart, use_container_width=True)
                    st.info("This chart shows the relationship between medication administration (red triangles/vertical lines) and vital sign measurements over time. Look for changes in vital signs following medication administration.")
                else:
                    st.warning("Not enough data to create a combined chart for the selected medication and vital sign.")
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_detailed_data(patient_data, tables):
    """Affiche les données détaillées pour un patient sous forme de texte clair et organisé"""
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.subheader("Detailed Patient Information")

    # regroupe les tables par catégorie
    table_categories = {
        "Medical Results": ["labevents", "microbiologyevents", "chartevents", "datetimeevents"],
        "Medications": ["emar", "emar_detail", "pharmacy", "prescriptions", "ingredientevents"],
        "Procedures": ["procedureevents", "procedures_icd", "hcpcsevents"],
        "Notes & Reports": ["noteevents", "radiology", "reports"],
        "Output & Intake": ["outputevents", "inputevents"],
        "Others": []
    }

    # classe les tables restantes
    for table in tables:
        if table not in sum(table_categories.values(), []):
            table_categories["Others"].append(table)

    # créer des onglets par catégorie
    categories = [cat for cat, tbls in table_categories.items() if any(t in patient_data for t in tbls)]
    if not categories:
        st.info("No detailed data available for this patient")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    category_tabs = st.tabs(categories)

    for i, category in enumerate(categories):
        with category_tabs[i]:
            for table in table_categories[category]:
                if table in patient_data and not patient_data[table].empty:
                    df = patient_data[table].drop(columns=['subject_id', 'hadm_id'], errors='ignore')

                    for col in df.select_dtypes(include=['datetime64', 'object']):
                        if 'date' in col.lower() or 'time' in col.lower():
                            try:
                                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M')
                            except:
                                pass

                    st.markdown(f"### {table.replace('_', ' ').title()}")
                    for idx, row in df.iterrows():
                        with st.expander(f"Record {idx + 1}", expanded=False):
                            for col, value in row.dropna().items():  # Supprime les valeurs vides
                                st.markdown(f"**{col.replace('_', ' ').title()}**: {value}")

    st.markdown('</div>', unsafe_allow_html=True)

def display_smart_patient_view(patient_data, tables):
    """Affiche les données patient de manière intelligente et contextuelle"""
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.subheader("Smart Patient View")
    
    # organisation par types d'informations cliniques
    clinical_data = {}
    
    # extraire les informations significatives par contexte
    if "labevents" in patient_data and not patient_data["labevents"].empty:
        lab_df = patient_data["labevents"]
        if "label" in lab_df.columns and "valuenum" in lab_df.columns and "charttime" in lab_df.columns:
            clinical_data["labs"] = lab_df[["charttime", "label", "valuenum", "valueuom"]].copy()
            clinical_data["labs"]["charttime"] = pd.to_datetime(clinical_data["labs"]["charttime"])
            clinical_data["labs"] = clinical_data["labs"].sort_values("charttime", ascending=False)
    
    if "chartevents" in patient_data and not patient_data["chartevents"].empty:
        chart_df = patient_data["chartevents"]
        if "label" in chart_df.columns and "valuenum" in chart_df.columns and "charttime" in chart_df.columns:
            clinical_data["vitals"] = chart_df[["charttime", "label", "valuenum", "valueuom"]].copy()
            clinical_data["vitals"]["charttime"] = pd.to_datetime(clinical_data["vitals"]["charttime"])
            clinical_data["vitals"] = clinical_data["vitals"].sort_values("charttime", ascending=False)
    
    if "emar" in patient_data and not patient_data["emar"].empty:
        med_df = patient_data["emar"]
        relevant_cols = [col for col in med_df.columns if col not in ["subject_id", "hadm_id"]]
        if relevant_cols:
            clinical_data["medications"] = med_df[relevant_cols].copy()
            if "charttime" in clinical_data["medications"].columns:
                clinical_data["medications"]["charttime"] = pd.to_datetime(clinical_data["medications"]["charttime"])
                clinical_data["medications"] = clinical_data["medications"].sort_values("charttime", ascending=False)
    
    if "procedures_icd" in patient_data and not patient_data["procedures_icd"].empty:
        proc_df = patient_data["procedures_icd"]
        relevant_cols = [col for col in proc_df.columns if col not in ["subject_id", "hadm_id"]]
        if relevant_cols:
            clinical_data["procedures"] = proc_df[relevant_cols].copy()
    
    if "diagnoses_icd" in patient_data and not patient_data["diagnoses_icd"].empty:
        diag_df = patient_data["diagnoses_icd"]
        relevant_cols = [col for col in diag_df.columns if col not in ["subject_id", "hadm_id"]]
        if relevant_cols:
            clinical_data["diagnoses"] = diag_df[relevant_cols].copy()
    
    # si aucune donnée n'est trouvée, afficher un message et retourner
    if not clinical_data:
        st.info("No structured clinical data found to display in the smart view")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # créer des onglets pour les différentes catégories de données
    clinical_categories = list(clinical_data.keys())
    cat_tabs = st.tabs([cat.title() for cat in clinical_categories])
    
    # afficher les données dans chaque onglet
    for i, category in enumerate(clinical_categories):
        with cat_tabs[i]:
            df = clinical_data[category]
            
            if category in ["labs", "vitals"]:
                # Grouper par type de mesure (label)
                unique_labels = df["label"].unique().tolist()
                
                # Option de filtrage
                selected_labels = st.multiselect(
                    "Filter measurements:", 
                    unique_labels,
                    default=unique_labels[:min(5, len(unique_labels))],
                    key=f"filter_{category}"
                )
                
                if not selected_labels:
                    st.warning("Please select at least one measurement to display")
                    continue
                    
                filtered_df = df[df["label"].isin(selected_labels)]
                
                # Afficher les dernières valeurs pour chaque mesure
                st.markdown("#### Latest Measurements")
                latest_values = filtered_df.sort_values("charttime", ascending=False).drop_duplicates("label")
                
                # Créer une disposition en colonnes pour les dernières valeurs
                cols = st.columns(min(3, len(latest_values)))
                for j, (_, row) in enumerate(latest_values.iterrows()):
                    col_idx = j % len(cols)
                    with cols[col_idx]:
                        unit = f" {row['valueuom']}" if pd.notna(row["valueuom"]) else ""
                        value = f"{row['valuenum']}{unit}" if pd.notna(row["valuenum"]) else "N/A"
                        st.metric(
                            label=row["label"], 
                            value=value,
                            delta_color="off",
                            help=f"Measured on {row['charttime'].strftime('%Y-%m-%d %H:%M')}"
                        )
                
                # Afficher l'historique sous forme de tableau
                st.markdown("#### Measurement History")
                history_df = filtered_df[["charttime", "label", "valuenum", "valueuom"]].sort_values("charttime", ascending=False)
                history_df.columns = ["Date/Time", "Measurement", "Value", "Unit"]
                
                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Date/Time": st.column_config.DatetimeColumn("Date/Time", format="DD/MM/YYYY HH:mm"),
                        "Measurement": st.column_config.TextColumn("Measurement"),
                        "Value": st.column_config.NumberColumn("Value"),
                        "Unit": st.column_config.TextColumn("Unit")
                    }
                )
                
            elif category == "medications":
                # Regrouper par médicament
                if "medication" in df.columns:
                    med_count = df["medication"].value_counts().reset_index()
                    med_count.columns = ["Medication", "Count"]
                    
                    st.markdown("#### Current Medications")
                    
                    if "dose_given" in df.columns and "dose_unit" in df.columns and "route" in df.columns:
                        # Créer un résumé des médicaments actuels
                        current_meds = df.sort_values("charttime", ascending=False).drop_duplicates("medication")
                        med_summary = pd.DataFrame({
                            "Medication": current_meds["medication"],
                            "Dose": current_meds.apply(lambda x: f"{x['dose_given']} {x['dose_unit']}" if pd.notna(x['dose_given']) else "N/A", axis=1),
                            "Route": current_meds["route"],
                            "Last Given": current_meds["charttime"].dt.strftime('%Y-%m-%d %H:%M')
                        })
                        
                        st.dataframe(
                            med_summary,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        # Afficher un simple comptage si les colonnes détaillées ne sont pas disponibles
                        st.bar_chart(med_count.set_index("Medication"))
                    
                    # Historique des administrations médicamenteuses
                    st.markdown("#### Medication Administration History")
                    med_history = df.sort_values("charttime", ascending=False)
                    
                    # Sélectionner les colonnes pertinentes pour l'affichage
                    display_cols = ["charttime", "medication"]
                    for col in ["dose_given", "dose_unit", "route", "scheduletime"]:
                        if col in med_history.columns:
                            display_cols.append(col)
                    
                    # Renommer les colonnes pour un affichage plus clair
                    col_names = {
                        "charttime": "Given Time",
                        "medication": "Medication",
                        "dose_given": "Dose",
                        "dose_unit": "Unit",
                        "route": "Route",
                        "scheduletime": "Scheduled Time"
                    }
                    
                    display_df = med_history[display_cols].rename(columns={col: col_names.get(col, col) for col in display_cols})
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("Medication data structure is not in the expected format")
            
            elif category in ["diagnoses", "procedures"]:
                # Pour les diagnostics et procédures, créer un affichage simple
                # On attend généralement des codes ICD avec descriptions
                relevant_cols = df.columns.tolist()
                
                # Vérifier si "icd_code" et "long_title" sont présents
                has_icd = "icd_code" in relevant_cols
                has_title = "long_title" in relevant_cols or "short_title" in relevant_cols
                
                if has_icd and has_title:
                    title_col = "long_title" if "long_title" in relevant_cols else "short_title"
                    
                    # Si sequencing est disponible, trier par seq_num
                    if "seq_num" in relevant_cols:
                        df = df.sort_values("seq_num")
                    
                    # Afficher sous forme de liste interactive
                    for _, row in df.iterrows():
                        with st.expander(f"{row[title_col]}", expanded=False):
                            for col in relevant_cols:
                                if col not in ["subject_id", "hadm_id"] and pd.notna(row[col]):
                                    st.markdown(f"**{col.replace('_', ' ').title()}**: {row[col]}")
                else:
                    # Affichage générique si les colonnes attendues ne sont pas présentes
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
            
            else:
                # Pour les autres catégories, affichage générique
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )
    
    st.markdown('</div>', unsafe_allow_html=True)

def load_patient_data(conn, tables, selected_patient, selected_admission):
    """charge les données du patient pour chaque table"""
    data = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tables = len(tables)
    for i, table in enumerate(tables):
        progress = (i) / total_tables
        progress_bar.progress(progress)
        status_text.text(f"Loading table {i+1}/{total_tables}: {table}")
        
        df = get_patient_data_by_admission(conn, table, selected_patient, selected_admission)
        data[table] = df
        
    progress_bar.progress(1.0)
    status_text.text("Loading complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    return data

def initialize_app():
    """initialize the application, establish database connection and get tables"""
    st.set_page_config(
        page_title="Patient Data Visualizer",
        page_icon="🏥",
        layout="wide"
    )
    
    # style
    st.markdown("""
    <style>
        .data-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header-style {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .stMetric label {
            font-weight: normal !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏥 Patient Data Visualizer")
    
    # connect to database
    conn = get_db_connection()
    
    # tables that have subject_id
    tables_with_subject_id = get_tables_with_subject_id(conn)
    
    return conn, tables_with_subject_id

def main():
    # initialiser l'état de session pour suivre l'état de sélection
    if 'patient_selected' not in st.session_state:
        st.session_state.patient_selected = False
    if 'admission_selected' not in st.session_state:
        st.session_state.admission_selected = False
    
    # config de la page
    conn, tables = initialize_app()
    
    if not tables:
        st.error("No tables with patient IDs found in the database!")
        return
    
    # récupérer les id (1ère table, limité à 1000)
    patient_ids = get_patient_ids(conn, tables[0])
    
    # créer une page avec une partie gauche et une partie droite
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.subheader("Patient Selection")
        
        # zone de recherche pour filtrer les patients
        search_id = st.text_input("Search Patient ID:", "")
        
        filtered_ids = patient_ids
        if search_id:
            filtered_ids = [pid for pid in patient_ids if str(search_id) in str(pid)]
        
        if not filtered_ids:
            st.warning("No matching patient IDs found.")
            filtered_ids = patient_ids
        
        selected_patient = st.selectbox(
            "Patient ID:",
            filtered_ids,
            key="patient_selector"
        )
        
        # bouton pour valider la sélection du patient
        select_patient_button = st.button("Select Patient", key="select_patient")
        
        if select_patient_button:
            st.session_state.patient_selected = True
            st.session_state.admission_selected = False  # réinitialiser la sélection d'admission
        
        # afficher la liste d'admission uniquement si un patient a été sélectionné
        if st.session_state.patient_selected:
            st.divider()
            st.subheader("Admission Selection")
        
            # récupérer les admissions
            admissions_df = get_admission_ids(conn, selected_patient)
            
            if not admissions_df.empty:
                admission_options = admissions_df['label'].tolist()
                admission_ids = admissions_df['hadm_id'].tolist()
                
                selected_admission_index = st.selectbox(
                    "Admission:",
                    range(len(admission_options)),
                    format_func=lambda i: admission_options[i],
                    key="admission_selector"
                )
                
                selected_admission = admission_ids[selected_admission_index]
                
                # afficher un résumé de l'admission sélectionnée
                admit_date = admissions_df['admittime'].iloc[selected_admission_index]
                discharge_date = admissions_df['dischtime'].iloc[selected_admission_index]
                duration = admissions_df['duration'].iloc[selected_admission_index]
                
                # stocker les dates
                st.session_state.admit_date = admit_date
                st.session_state.discharge_date = discharge_date
                
                st.markdown(f"""
                <div style='background-color: rgba(59, 113, 202, 0.1); padding: 10px; border-radius: 5px; border-left: 3px solid #3b71ca;'>
                    <strong>Duration:</strong> {duration:.1f} days<br>
                    <strong>From:</strong> {admit_date.strftime('%d/%m/%Y')}<br>
                    <strong>To:</strong> {discharge_date.strftime('%d/%m/%Y')}
                </div>
                """, unsafe_allow_html=True)
                
                # bouton pour charger les données basées sur l'admission sélectionnée
                load_button = st.button("Load Data", key="load_data")
                
                if load_button:
                    st.session_state.admission_selected = True
            else:
                st.warning("No admissions found for this patient.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # affiche les données du patient uniquement lorsque tout est sélectionné
    with col2:
        if st.session_state.patient_selected and st.session_state.admission_selected:
            # affiche un résumé du patient
            display_patient_summary(conn, selected_patient, selected_admission)
            
            # récupère et affiche les données de séries temporelles
            time_series_data = get_time_series_data(conn, selected_patient, selected_admission)
            display_time_series_data(
                time_series_data, 
                st.session_state.admit_date, 
                st.session_state.discharge_date
            )
            
            # charge les données détaillées du patient
            patient_data = load_patient_data(conn, tables, selected_patient, selected_admission)
            
            # Afficher la vue intelligente des données patient
            display_smart_patient_view(patient_data, tables)
            
            # Créer des onglets pour choisir entre l'affichage intelligent et l'affichage détaillé
            data_view_tabs = st.tabs(["Detailed Data View"])
            
            with data_view_tabs[0]:
                # Afficher les données détaillées améliorées
                display_detailed_data(patient_data, tables)
            
        elif st.session_state.patient_selected:
            st.info("👈 Please select an admission and click 'Load Data'.")
        else:
            st.info("👈 Please select a patient ID first.")

if __name__ == "__main__":
    main()
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
        # Utilisation explicite de l'index sur subject_id
        query = f"""
        SELECT DISTINCT subject_id
        FROM [{table}]
        ORDER BY subject_id  /* L'index sera utilisé pour l'ordre */
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
            WHERE subject_id = ? AND hadm_id = ?  /* L'ordre subject_id puis hadm_id est important */
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
def get_standardized_medications(_conn, subject_id, hadm_id, source='emar'):
    """Récupère et standardise les données de médicaments"""
    if source == 'emar':
        query = """
        SELECT charttime, medication, scheduletime, storetime, event_txt
        FROM emar
        WHERE subject_id = ? AND hadm_id = ?  /* Ordre optimal pour utiliser l'index */
        ORDER BY charttime
        """
    else:
        return pd.DataFrame()
    
    try:
        df = pd.read_sql_query(query, _conn, params=[subject_id, hadm_id])
        if not df.empty:
            df['charttime'] = pd.to_datetime(df['charttime'])
            df['scheduletime'] = pd.to_datetime(df['scheduletime'])
            
            # Standardisation des noms de médicaments
            df['medication'] = df['medication'].str.title()
            
            return df
    except Exception as e:
        st.error(f"Medication retrieval error: {e}")
    
    return pd.DataFrame()

def create_medication_filters(emar_meds, prefix, create_columns=True, suffix=""):
    """Crée des filtres pour les médicaments sans filtre par event_txt"""
    
    # Générer une clé unique en combinant prefix et suffix
    key_suffix = f"_{suffix}" if suffix else ""
    key = f"{prefix}_medication_selector{key_suffix}"
    
    # Extraire les noms de médicaments uniques
    unique_meds = emar_meds['medication'].unique()
    
    if create_columns:
        try:
            # Créer les colonnes
            columns = st.columns([1, 1])
            
            # Utiliser les colonnes
            selected_med = columns[0].selectbox(
                "Select medication",
                options=[""] + list(unique_meds),
                key=key
            )
        except Exception as e:
            st.error(f"Error creating columns: {e}")
            # Fallback sans colonnes
            selected_med = st.selectbox(
                "Select medication",
                options=[""] + list(unique_meds),
                key=key
            )
    else:
        # Version sans colonnes
        selected_med = st.selectbox(
            "Select medication",
            options=[""] + list(unique_meds),
            key=key
        )
    
    return selected_med

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
        SELECT charttime, medication, scheduletime, storetime, event_txt
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
def get_standardized_vitals(_conn, subject_id, hadm_id):
    """Récupère et standardise les données de signes vitaux"""
    query = """
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
        df = pd.read_sql_query(query, _conn, params=[subject_id, hadm_id])
    except Exception as e:
        st.error(f"Error during vital signs recovery: {e}")
    
    return pd.DataFrame()

def create_vitals_filters(vital_data, prefix, create_columns=True):
    """Crée des filtres pour les signes vitaux"""
    
    unique_vitals = vital_data['label'].unique()
    
    if create_columns:
        col1, col2 = st.columns([1, 1])
        with col1:
            selected_vital = st.selectbox(
                "Select vital sign",
                options=[""] + list(unique_vitals),
                key=f"{prefix}_vital_selector"
            )
    else:
        selected_vital = st.selectbox(
            "Select vital sign",
            options=[""] + list(unique_vitals),
            key=f"{prefix}_vital_selector"
        )
    
    return selected_vital

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
    
# Dans la fonction create_single_medication_chart, remplacez les lignes problématiques 
# où vous essayez d'additionner des objets Timestamp

# Version incorrecte:
# filtered_data['storetime'] = filtered_data['storetime'].fillna(
#     filtered_data['scheduletime'] + pd.Timedelta(hours=1))

# Version corrigée:
def create_single_medication_chart(emar_meds, selected_med, start_date=None, end_date=None):
    """Crée un graphique pour un médicament spécifique"""
    if not selected_med:
        return None
    
    # Utiliser selected_med comme medication_name pour l'affichage
    medication_name = selected_med
    
    # Filtrer les données pour le médicament sélectionné
    filtered_data = emar_meds[emar_meds['medication'] == selected_med].copy()
    
    # Si aucune donnée après filtrage par médicament, retourner None
    if filtered_data.empty:
        return None
    
    # Convertir les colonnes de dates en datetime avant le filtrage
    datetime_columns = ['charttime', 'scheduletime', 'storetime']
    for col in datetime_columns:
        if col in filtered_data.columns:
            filtered_data[col] = pd.to_datetime(filtered_data[col], errors='coerce')
    
    # Filtrage par date si spécifié
    if start_date and end_date:
        # Convertir start_date et end_date en Timestamp si ce ne sont pas des objets datetime
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.to_datetime(start_date)
        if not isinstance(end_date, pd.Timestamp):
            end_date = pd.to_datetime(end_date)
        
        # Utiliser scheduletime pour le filtrage
        if 'scheduletime' in filtered_data.columns:
            date_filtered = filtered_data[
                (filtered_data['scheduletime'] >= start_date) & 
                (filtered_data['scheduletime'] <= end_date)
            ].copy()
            
            # Si le filtrage par scheduletime ne retourne rien, essayer avec charttime
            if date_filtered.empty and 'charttime' in filtered_data.columns:
                date_filtered = filtered_data[
                    (filtered_data['charttime'] >= start_date) & 
                    (filtered_data['charttime'] <= end_date)
                ].copy()
            
            filtered_data = date_filtered
    
    # Si aucune donnée après filtrage par date, retourner None
    if filtered_data.empty:
        return None
    
    # Remplir les valeurs manquantes dans scheduletime et storetime
    if 'scheduletime' not in filtered_data.columns and 'charttime' in filtered_data.columns:
        filtered_data['scheduletime'] = filtered_data['charttime']
    
    if 'storetime' not in filtered_data.columns:
        if 'charttime' in filtered_data.columns:
            # Ajouter une heure à charttime
            filtered_data['storetime'] = filtered_data['charttime'].apply(
                lambda x: x + pd.Timedelta(hours=1) if pd.notnull(x) else None
            )
        elif 'scheduletime' in filtered_data.columns:
            # Ajouter une heure à scheduletime
            filtered_data['storetime'] = filtered_data['scheduletime'].apply(
                lambda x: x + pd.Timedelta(hours=1) if pd.notnull(x) else None
            )
    
    # Remplir les NaN
    if 'scheduletime' in filtered_data.columns and 'charttime' in filtered_data.columns:
        # Remplacer les NaN dans scheduletime par charttime
        filtered_data.loc[filtered_data['scheduletime'].isna(), 'scheduletime'] = \
            filtered_data.loc[filtered_data['scheduletime'].isna(), 'charttime']
    
    if 'storetime' in filtered_data.columns and 'scheduletime' in filtered_data.columns:
        # Remplacer les NaN dans storetime
        mask = filtered_data['storetime'].isna()
        filtered_data.loc[mask, 'storetime'] = filtered_data.loc[mask, 'scheduletime'].apply(
            lambda x: x + pd.Timedelta(hours=1) if pd.notnull(x) else None
        )
    
    # Éliminer les lignes avec des NaN dans scheduletime ou storetime
    filtered_data = filtered_data.dropna(subset=['scheduletime', 'storetime'])
    
    # Vérifier encore une fois si des données restent
    if filtered_data.empty:
        return None
    
    med_data = filtered_data
    
    # Set date range for the chart
    min_date = start_date if start_date else med_data['scheduletime'].min()
    max_date = end_date if end_date else med_data['storetime'].max()
    
    # Create figure
    fig = go.Figure()
    
    # Sort by scheduletime to process overlaps
    med_data = med_data.sort_values('scheduletime')
    
    # Process each medication administration and handle overlaps
    row_positions = {}  # Track vertical positions for each row
    
    for idx, row in med_data.iterrows():
        start_time = row['scheduletime']
        end_time = row['storetime']
        
        if pd.isnull(start_time) or pd.isnull(end_time):
            continue
            
        # Make sure end time is after start time
        if end_time <= start_time:
            # Ajouter une heure pour avoir une durée valide
            end_time = start_time + pd.Timedelta(hours=1)
            
        # Determine vertical position (row) for this administration
        row_level = 0
        while True:
            overlap = False
            if row_level in row_positions:
                for pos_start, pos_end in row_positions[row_level]:
                    if max(start_time, pos_start) <= min(end_time, pos_end):
                        overlap = True
                        break
            if not overlap:
                break
            row_level += 1
        
        # Store position
        if row_level not in row_positions:
            row_positions[row_level] = []
        row_positions[row_level].append((start_time, end_time))
        
        # Définir la couleur en fonction de event_txt
        color = 'rgb(128, 128, 128)'  # Gris par défaut
        if 'event_txt' in row and pd.notna(row['event_txt']):
            if row['event_txt'].lower() in ['not given', 'canceled', 'cancelled']:
                color = 'rgb(255, 0, 0)'  # Rouge
            elif row['event_txt'].lower() in ['given', 'administered', 'flushed']:
                color = 'rgb(0, 128, 0)'  # Vert
        
        # Add horizontal line for the administration duration
        fig.add_shape(
            type="line",
            x0=start_time,
            y0=row_level + 1,
            x1=end_time,
            y1=row_level + 1,
            line=dict(
                color=color,
                width=4,
            )
        )
        
        # Add marker at start for better visibility
        fig.add_trace(go.Scatter(
            x=[start_time],
            y=[row_level + 1],
            mode='markers',
            marker=dict(
                color=color,
                size=8,
                symbol='circle',
            ),
            showlegend=False,
            hoverinfo='text',
            hovertext=f"Start: {start_time.strftime('%Y-%m-%d %H:%M')}"
        ))
        
        # Build hover information text
        dose_info = f"Dose: {row['dose_given']} {row['dose_unit']}" if 'dose_given' in row and pd.notna(row['dose_given']) else ""
        route_info = f"Route: {row['route']}" if 'route' in row and pd.notna(row['route']) else ""
        event_info = f"Status: {row['event_txt']}" if 'event_txt' in row and pd.notna(row['event_txt']) else ""
        
        hover_text = f"{medication_name}<br>Schedule: {start_time.strftime('%Y-%m-%d %H:%M')}<br>End: {end_time.strftime('%Y-%m-%d %H:%M')}"
        if event_info:
            hover_text += f"<br>{event_info}"
        if dose_info:
            hover_text += f"<br>{dose_info}"
        if route_info:
            hover_text += f"<br>{route_info}"
            
        # Add invisible trace for hover information
        fig.add_trace(go.Scatter(
            x=[(start_time + (end_time - start_time)/2)],  # Middle of line - fixed calculation
            y=[row_level + 1],
            mode='markers',
            marker=dict(
                color='rgba(0,0,0,0)',  # Invisible
                size=1,
            ),
            showlegend=False,
            hoverinfo='text',
            hovertext=hover_text
        ))
    
    # Update layout
    max_rows = max(row_positions.keys()) + 2 if row_positions else 2
    
    fig.update_layout(
        title=f"{medication_name} Administration Timeline",
        xaxis_title="Date/Time",
        yaxis=dict(
            title="Administrations",
            range=[0, max_rows],
            tickvals=list(range(1, max_rows)),
            ticktext=[f"Admin {i+1}" for i in range(max_rows-1)],
        ),
        height=max(300, 50 + 40 * max_rows),  # Adjust height based on number of rows
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis_range=[min_date, max_date],
        plot_bgcolor='rgba(248,249,250,0.5)',
        title_font=dict(size=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
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
    st.subheader("Medication Administration History")
    
    if 'medication' in medication_data.columns and 'charttime' in medication_data.columns:
        # Colonnes à afficher
        display_cols = ['charttime', 'medication']
        for col in ['scheduletime', 'storetime', 'event_txt']:
            if col in medication_data.columns:
                display_cols.append(col)
        
        # Créer le dataframe pour l'affichage
        med_history = medication_data[display_cols].sort_values('charttime', ascending=False)
        
        # Renommer les colonnes pour un affichage plus clair
        column_mapping = {
            'charttime': 'Administration Time',
            'medication': 'Medication',
            'scheduletime': 'Scheduled Time',
            'storetime': 'Store Time',
            'event_txt': 'Event Type'
        }
        
        med_history = med_history.rename(columns={col: column_mapping.get(col, col) for col in med_history.columns})
        
        # Filtrer par médicament spécifique si demandé
        available_meds = medication_data['medication'].unique().tolist()
        med_filter = st.checkbox("Filter history by medication", value=False, key="med_history_filter")
        if med_filter:
            selected_med_history = st.selectbox(
                "Select medication:",
                options=available_meds,
                key="med_history_selection"
            )
            if selected_med_history:
                med_history = med_history[med_history['Medication'] == selected_med_history]

        # Afficher le dataframe filtré
        st.dataframe(med_history, use_container_width=True, hide_index=True)

        st.markdown('</div>', unsafe_allow_html=True)
                

def display_patient_summary(conn, subject_id, hadm_id):
    """display summary of patient information"""
    demographics = get_patient_demographics(conn, subject_id)
    
    # admission - optimisé pour utiliser l'index
    admission_query = """
    SELECT admission_type, admission_location, discharge_location, 
           admittime, dischtime, insurance, language, marital_status
    FROM admissions
    WHERE subject_id = ? AND hadm_id = ?  /* L'ordre subject_id puis hadm_id est important */
    """
    admission = pd.read_sql_query(admission_query, conn, params=[subject_id, hadm_id])
    
    # diagnoses - optimisé
    diagnostics_query = """
    SELECT d.icd_code, i.long_title
    FROM diagnoses_icd d
    JOIN d_icd_diagnoses i ON d.icd_code = i.icd_code AND d.icd_version = i.icd_version
    WHERE d.subject_id = ? AND d.hadm_id = ?  /* Ordre optimal pour utiliser l'index */
    ORDER BY d.seq_num
    """
    try:
        diagnostics = pd.read_sql_query(diagnostics_query, conn, params=[subject_id, hadm_id])
    except:
        diagnostics = pd.DataFrame()
    
    # procedures - optimisé
    procedures_query = """
    SELECT p.icd_code, i.long_title
    FROM procedures_icd p
    JOIN d_icd_procedures i ON p.icd_code = i.icd_code AND p.icd_version = i.icd_version
    WHERE p.subject_id = ? AND p.hadm_id = ?  /* Ordre optimal pour utiliser l'index */
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
    
    # Date par défaut = plage complète (admission à sortie)
    default_start_date = admit_time
    default_end_date = discharge_time
    
    # Initialiser une clé de session pour suivre si l'utilisateur a modifié le slider
    if 'date_slider_changed' not in st.session_state:
        st.session_state.date_slider_changed = False
    
    # date range slider for all charts
    dates = st.slider(
        "Select Date Range for Charts (or use full admission period)",
        min_value=admit_time.to_pydatetime(),
        max_value=discharge_time.to_pydatetime(),
        value=(admit_time.to_pydatetime(), discharge_time.to_pydatetime()),
        format="DD/MM/YY HH:mm",
        key="date_range_slider"
    )
    
    # Si l'utilisateur a modifié le slider, utiliser les dates sélectionnées
    # Sinon, utiliser la période d'admission complète
    if st.session_state.date_slider_changed:
        selected_start_date, selected_end_date = dates
    else:
        # Vérifier si le slider est différent des valeurs par défaut
        if dates[0] != admit_time.to_pydatetime() or dates[1] != discharge_time.to_pydatetime():
            st.session_state.date_slider_changed = True
            selected_start_date, selected_end_date = dates
        else:
            selected_start_date, selected_end_date = default_start_date, default_end_date
    
    # Ajouter un bouton pour réinitialiser aux dates d'admission complètes
    if st.button("Reset to Full Admission Period", key="reset_date_range"):
        st.session_state.date_slider_changed = False
        # On doit rafraîchir la page pour réinitialiser le slider
        st.experimental_rerun()
    
    # Ajout d'un onglet pour organiser les visualisations
    viz_tabs = st.tabs(["Labs", "Vital Signs", "Medications", "Individual Medication", "Correlations"])
    
    # Onglet 1: Labos
    with viz_tabs[0]:
        if 'labevents' in time_series_data and not time_series_data['labevents'].empty:
            lab_chart = create_time_series_chart(
                time_series_data['labevents'],
                "Lab Results",
                "Value",
                selected_start_date,
                selected_end_date
            )
            if lab_chart:
                st.plotly_chart(lab_chart, use_container_width=True)
    
    # Onglet 2: Signes vitaux
    with viz_tabs[1]:
        if 'chartevents' in time_series_data and not time_series_data['chartevents'].empty:
            vitals_chart = create_time_series_chart(
                time_series_data['chartevents'],
                "Vital Signs",
                "Value",
                selected_start_date,
                selected_end_date
            )
            if vitals_chart:
                st.plotly_chart(vitals_chart, use_container_width=True)
    
    # Onglet 3: Médicaments
    with viz_tabs[2]:
        if 'medications' in time_series_data and not time_series_data['medications'].empty:
            emar_meds = time_series_data['medications']
            med_chart = create_medication_timeline(
                emar_meds,
                selected_start_date,
                selected_end_date
            )
            if med_chart:
                st.plotly_chart(med_chart, use_container_width=True)
    
    # Onglet 4: Médicament individuel
    with viz_tabs[3]:
        if 'medications' in time_series_data and not time_series_data['medications'].empty:
            emar_meds = time_series_data['medications']
            st.markdown("### Chronology of medication administration")
            
            if not emar_meds.empty:
                # Utilisation de la fonction standardisée pour créer des filtres cohérents
                selected_med = create_medication_filters(emar_meds, "corr", create_columns=False, suffix="med_corr")
                
                if selected_med:                    
                    # Créer le graphique avec gestion améliorée des erreurs
                    single_med_chart = create_single_medication_chart(
                        emar_meds,
                        selected_med,
                        start_date=selected_start_date,
                        end_date=selected_end_date
                    )
                    
                    if single_med_chart:
                        st.plotly_chart(single_med_chart, use_container_width=True)
                    else:
                        st.warning(f"No data available for {selected_med} with the selected filters")
            else:
                st.warning("No medication available")
    
    # Onglet 5: Corrélations médicament-vital
    with viz_tabs[4]:
        if ('medications' in time_series_data and not time_series_data['medications'].empty and
            'chartevents' in time_series_data and not time_series_data['chartevents'].empty):
            
            emar_meds = time_series_data['medications']
            vital_data = time_series_data['chartevents']
            
            st.markdown("### Drug-Vital Sign Correlation")
            
            # Appel direct sans créer de colonnes supplémentaires
            selected_med = create_medication_filters(emar_meds, "corr", create_columns=False, suffix="vital_corr")
            selected_vital = create_vitals_filters(vital_data, "corr", create_columns=False)
            
            combined_chart = create_combined_medication_vital_chart(
                emar_meds,
                vital_data,
                selected_med,
                selected_vital,
                selected_start_date,
                selected_end_date
            )
            
            if combined_chart:
                st.plotly_chart(combined_chart, use_container_width=True)
            else:
                st.warning("Not enough data available")
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_detailed_data(patient_data, tables):
    """Affiche les données détaillées pour un patient avec filtrage"""
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.subheader("Detailed Patient Information")

    # Regroupe les tables par catégorie
    table_categories = {
        "Medical Results": ["labevents", "microbiologyevents", "chartevents", "datetimeevents"],
        "Medications": ["emar", "emar_detail", "pharmacy", "prescriptions", "ingredientevents"],
        "Procedures": ["procedureevents", "procedures_icd", "hcpcsevents"],
        "Notes & Reports": ["noteevents", "radiology", "reports"],
        "Output & Intake": ["outputevents", "inputevents"],
        "Others": []
    }

    # Classe les tables restantes
    for table in tables:
        if table not in sum(table_categories.values(), []):
            table_categories["Others"].append(table)

    # Créer des onglets par catégorie
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
                    
                    # Ajout d'un filtre de recherche par texte
                    search_term = st.text_input(f"Filter records in {table.title()} (type to search in any field):", key=f"filter_{table}")
                    
                    # Ajout d'un filtre par colonne
                    if len(df.columns) > 0:
                        filter_col = st.selectbox(f"Filter by column:", ["None"] + list(df.columns), key=f"filter_col_{table}")
                        
                        if filter_col != "None":
                            unique_values = df[filter_col].dropna().unique()
                            if len(unique_values) <= 30:  # N'affiche le sélecteur que si le nombre de valeurs est raisonnable
                                filter_value = st.multiselect(f"Select values for {filter_col}:", 
                                                             options=unique_values, 
                                                             key=f"filter_val_{table}")
                                if filter_value:
                                    df = df[df[filter_col].isin(filter_value)]
                    
                    # Application du filtre de recherche textuelle
                    if search_term:
                        mask = pd.Series(False, index=df.index)
                        for col in df.columns:
                            mask |= df[col].astype(str).str.contains(search_term, case=False, na=False)
                        df = df[mask]
                    
                    # Affichage des résultats filtrés
                    if not df.empty:
                        st.write(f"Showing {len(df)} record(s)")
                        
                        # Limiter le nombre d'enregistrements affichés si trop nombreux
                        max_records_to_display = 30
                        if len(df) > max_records_to_display:
                            st.warning(f"Displaying first {max_records_to_display} records. Use filters to narrow down results.")
                            df = df.head(max_records_to_display)
                        
                        for idx, row in df.iterrows():
                            with st.expander(f"Record {idx + 1}", expanded=False):
                                for col, value in row.dropna().items():  # Supprime les valeurs vides
                                    st.markdown(f"**{col.replace('_', ' ').title()}**: {value}")
                    else:
                        st.info("No records match the filter criteria.")

    st.markdown('</div>', unsafe_allow_html=True)


def display_smart_patient_view(patient_data, tables):
    """Affiche uniquement les données de médicaments de manière intelligente"""
    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.subheader("Smart Patient View")
    
    # On ne garde que les médicaments
    clinical_data = {}
    
    if "emar" in patient_data and not patient_data["emar"].empty:
        med_df = patient_data["emar"]
        relevant_cols = [col for col in med_df.columns if col not in ["subject_id", "hadm_id"]]
        if relevant_cols:
            clinical_data["medications"] = med_df[relevant_cols].copy()
            if "charttime" in clinical_data["medications"].columns:
                clinical_data["medications"]["charttime"] = pd.to_datetime(clinical_data["medications"]["charttime"])
                clinical_data["medications"] = clinical_data["medications"].sort_values("charttime", ascending=False)
    
    # Si aucune donnée n'est trouvée, afficher un message et retourner
    if not clinical_data or "medications" not in clinical_data:
        st.info("No medication data found to display in the smart view")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Afficher uniquement les médicaments
    df = clinical_data["medications"]
    
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
        for col in ["dose_given", "dose_unit", "route", "scheduletime", "event_txt"]:
            if col in med_history.columns:
                display_cols.append(col)
        
        # Renommer les colonnes pour un affichage plus clair
        col_names = {
            "charttime": "Given Time",
            "medication": "Medication",
            "dose_given": "Dose",
            "dose_unit": "Unit",
            "route": "Route",
            "scheduletime": "Scheduled Time",
            "event_txt": "Event Status"
        }
        
        display_df = med_history[display_cols].rename(columns={col: col_names.get(col, col) for col in display_cols})
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

def load_patient_data_optimized(conn, tables, selected_patient, selected_admission):
    """charge les données du patient en parallèle pour chaque table"""
    data = {}
    
    # Liste des tables les plus importantes à charger en priorité
    priority_tables = [
        "admissions", "patients", "emar", "labevents", 
        "chartevents", "diagnoses_icd", "procedures_icd"
    ]
    
    # Réorganiser les tables pour traiter d'abord les prioritaires
    ordered_tables = [t for t in priority_tables if t in tables]
    ordered_tables += [t for t in tables if t not in priority_tables]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tables = len(ordered_tables)
    for i, table in enumerate(ordered_tables):
        progress = (i) / total_tables
        progress_bar.progress(progress)
        status_text.text(f"Loading table {i+1}/{total_tables}: {table}")
        
        df = get_patient_data_by_admission(conn, table, selected_patient, selected_admission)
        if not df.empty:  # Ne stocke que les tables non vides
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
            patient_data = load_patient_data_optimized(conn, tables, selected_patient, selected_admission)
            
            # Obtenez les données de médicaments et de signes vitaux pour le dashboard
            medication_data = get_standardized_medications(conn, selected_patient, selected_admission)
            vital_signs_data = get_standardized_vitals(conn, selected_patient, selected_admission)

            # Affichez uniquement le dashboard de médicaments
            display_medication_dashboard(medication_data, vital_signs_data, 
                                    st.session_state.admit_date, 
                                    st.session_state.discharge_date)
            
            # Créer des onglets pour choisir entre l'affichage intelligent et l'affichage détaillé
            data_view_tabs = st.tabs(["Detailed Data View", "Smart Patient View"])

            with data_view_tabs[0]:
                # Afficher les données détaillées améliorées
                display_detailed_data(patient_data, tables)
                
            with data_view_tabs[1]:
                # Afficher la vue intelligente des données patient
                display_smart_patient_view(patient_data, tables)
            
        elif st.session_state.patient_selected:
            st.info("👈 Please select an admission and click 'Load Data'.")
        else:
            st.info("👈 Please select a patient ID first.")

if __name__ == "__main__":
    main()

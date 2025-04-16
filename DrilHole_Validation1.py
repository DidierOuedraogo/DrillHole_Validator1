import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
import base64
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Validation de Données de Forage Minier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fonction pour ajouter un style CSS personnalisé
def add_custom_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f5f7f9;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .block-container {
            padding: 2rem;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stAlert {
            border-radius: 8px;
        }
        .stButton>button {
            border-radius: 5px;
            background-color: #4b7bec;
            color: white;
            font-weight: 500;
            border: none;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #3867d6;
        }
        .dataframe {
            font-size: 14px;
        }
        .css-1d391kg {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
        }
        .mapping-box {
            background-color: #f0f5ff;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid #4b7bec;
        }
    </style>
    """, unsafe_allow_html=True)

add_custom_css()

# Titre et description de l'application
st.title('🔍 Validation de Données de Forage Minier')
st.markdown("""
Cette application permet de valider les données de forage d'exploration minière.
Importez vos fichiers CSV ou Excel et effectuez diverses vérifications pour assurer la qualité et la cohérence des données.

**Développé par:** Didier Ouedraogo, P.Geo
""")

# Initialisation des variables de session
if 'files' not in st.session_state:
    st.session_state.files = {}
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = {}
if 'composite_data' not in st.session_state:
    st.session_state.composite_data = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}

# Fonction pour charger un fichier
def load_file(file, file_type):
    try:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            st.error(f"Format de fichier non pris en charge pour {file.name}. Utilisez CSV ou Excel.")
            return None
        
        # Standardisation des noms de colonnes (mise en minuscule)
        data.columns = [col.lower() for col in data.columns]
        
        # Vérification des colonnes essentielles pour chaque type de fichier
        if file_type == 'collars' and not all(col in data.columns for col in ['holeid', 'depth']):
            st.error("Le fichier collars doit contenir au moins les colonnes 'holeid' et 'depth'.")
            return None
        elif file_type == 'survey' and not all(col in data.columns for col in ['holeid', 'depth']):
            st.error("Le fichier survey doit contenir au moins les colonnes 'holeid' et 'depth'.")
            return None
        elif file_type in ['assays', 'litho', 'density', 'oxidation', 'geometallurgy'] and not all(col in data.columns for col in ['holeid', 'from', 'to']):
            st.error(f"Le fichier {file_type} doit contenir au moins les colonnes 'holeid', 'from' et 'to'.")
            return None
        
        st.success(f"Fichier {file_type} chargé avec succès: {file.name}")
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier {file.name}: {str(e)}")
        return None

# Fonction pour vérifier les forages manquants
def check_missing_holes(collar_data, other_data, file_type):
    collar_holes = set(collar_data['holeid'].unique())
    other_holes = set(other_data['holeid'].unique())
    
    # Forages dans collar mais absents dans l'autre fichier
    missing_in_other = collar_holes - other_holes
    
    # Forages dans l'autre fichier mais absents dans collar
    extra_in_other = other_holes - collar_holes
    
    return {
        'missing_in_other': list(missing_in_other),
        'extra_in_other': list(extra_in_other)
    }

# Fonction pour vérifier les profondeurs des intervalles
def check_interval_depths(collar_data, interval_data, file_type):
    results = []
    
    # Créer un dictionnaire des profondeurs maximales pour chaque forage
    max_depths = dict(zip(collar_data['holeid'], collar_data['depth']))
    
    # Vérifier chaque intervalle
    for idx, row in interval_data.iterrows():
        hole_id = row['holeid']
        from_depth = row['from']
        to_depth = row['to']
        
        # Vérifier si le forage existe dans collar
        if hole_id in max_depths:
            max_depth = max_depths[hole_id]
            
            # Vérifier si les profondeurs sont valides
            if to_depth > max_depth:
                results.append({
                    'holeid': hole_id,
                    'from': from_depth,
                    'to': to_depth,
                    'max_depth': max_depth,
                    'issue': f"La profondeur 'to' ({to_depth}) dépasse la profondeur maximale du forage ({max_depth})"
                })
            
            if from_depth > to_depth:
                results.append({
                    'holeid': hole_id,
                    'from': from_depth,
                    'to': to_depth,
                    'max_depth': max_depth,
                    'issue': f"La profondeur 'from' ({from_depth}) est supérieure à 'to' ({to_depth})"
                })
    
    return pd.DataFrame(results) if results else pd.DataFrame()

# Fonction pour vérifier les doublons
def check_duplicates(data, file_type):
    if file_type == 'collars':
        # Vérifier les doublons de forages dans collars
        duplicates = data[data.duplicated(subset=['holeid'], keep=False)]
        return duplicates
    elif file_type in ['assays', 'litho', 'density', 'oxidation', 'geometallurgy']:
        # Vérifier les doublons d'intervalles
        duplicates = data[data.duplicated(subset=['holeid', 'from', 'to'], keep=False)]
        return duplicates
    elif file_type == 'survey':
        # Vérifier les doublons dans les mesures de survey
        duplicates = data[data.duplicated(subset=['holeid', 'depth'], keep=False)]
        return duplicates
    else:
        return pd.DataFrame()

# Fonction améliorée pour vérifier les échantillons composites proches
def check_nearby_composites(composite_data, distance_threshold=0.1, column_mapping=None):
    # Appliquer le mapping de colonnes si fourni
    if column_mapping and all(k in column_mapping for k in ['x', 'y', 'z']):
        temp_data = composite_data.copy()
        for target_col, source_col in column_mapping.items():
            if source_col in composite_data.columns:
                temp_data[target_col] = temp_data[source_col]
        composite_data = temp_data
    
    # Vérifier si les colonnes de coordonnées existent
    required_columns = ['x', 'y', 'z']
    missing_columns = [col for col in required_columns if col not in composite_data.columns]
    
    if missing_columns:
        return {'status': 'error', 'message': f"Colonnes manquantes dans les données de composite: {', '.join(missing_columns)}"}
    
    # Vérifier si les colonnes contiennent des données numériques
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(composite_data[col]):
            try:
                # Essayer de convertir en nombre
                composite_data[col] = pd.to_numeric(composite_data[col])
            except Exception as e:
                return {'status': 'error', 'message': f"Échec de conversion de la colonne '{col}' en type numérique: {str(e)}"}
    
    # Vérifier les valeurs manquantes
    missing_values = composite_data[required_columns].isna().sum().sum()
    if missing_values > 0:
        composite_data = composite_data.dropna(subset=required_columns)
    
    # Extraire les coordonnées
    coords = composite_data[required_columns].values
    
    # Vérifier qu'il reste suffisamment de points
    if len(coords) < 2:
        return {'status': 'error', 'message': "Pas assez de points avec des coordonnées valides pour effectuer l'analyse."}
    
    # Utiliser KNN pour trouver les points proches
    try:
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
    except Exception as e:
        return {'status': 'error', 'message': f"Erreur lors de l'analyse des voisins les plus proches: {str(e)}"}
    
    # Filtrer les points dont la distance est inférieure au seuil
    close_points = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        # idx[0] est le point lui-même, idx[1] est le voisin le plus proche
        if dist[1] < distance_threshold:
            close_points.append({
                'point1_index': i,
                'point2_index': idx[1],
                'distance': dist[1],
                'point1_holeid': composite_data.iloc[i]['holeid'] if 'holeid' in composite_data.columns else f"Point_{i}",
                'point2_holeid': composite_data.iloc[idx[1]]['holeid'] if 'holeid' in composite_data.columns else f"Point_{idx[1]}",
                'point1_coords': coords[i],
                'point2_coords': coords[idx[1]]
            })
    
    # Retourner les résultats avec un statut
    if close_points:
        return {'status': 'success', 'points': close_points, 'count': len(close_points)}
    else:
        return {'status': 'empty', 'message': f"Aucun point proche trouvé avec un seuil de {distance_threshold} m"}

# Fonction pour exporter les résultats
def export_to_excel(validation_results):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    for result_name, result_data in validation_results.items():
        if isinstance(result_data, pd.DataFrame) and not result_data.empty:
            result_data.to_excel(writer, sheet_name=result_name[:31])  # Excel limite à 31 caractères
        elif isinstance(result_data, dict):
            # Pour les résultats de forages manquants
            missing_df = pd.DataFrame({
                'forage': result_data.get('missing_in_other', []),
                'status': ['Manquant dans fichier secondaire'] * len(result_data.get('missing_in_other', []))
            })
            extra_df = pd.DataFrame({
                'forage': result_data.get('extra_in_other', []),
                'status': ['Présent dans fichier secondaire mais absent dans collars'] * len(result_data.get('extra_in_other', []))
            })
            pd.concat([missing_df, extra_df]).to_excel(writer, sheet_name=result_name[:31])
    
    writer.close()  # Remplace writer.save() qui est déprécié
    processed_data = output.getvalue()
    return processed_data

# Fonction pour créer un lien de téléchargement
def get_download_link(data, filename):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Télécharger le fichier</a>'
    return href

# Interface utilisateur avec onglets
tab1, tab2, tab3 = st.tabs(["Import de données", "Validation de données", "Composites"])

with tab1:
    st.header("Import de fichiers de forage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fichier des collars (obligatoire)")
        collar_file = st.file_uploader("Sélectionner le fichier collars (CSV, Excel)", type=["csv", "xlsx", "xls"], key="collar_uploader")
        if collar_file is not None:
            collar_data = load_file(collar_file, 'collars')
            if collar_data is not None:
                st.session_state.files['collars'] = collar_data
                st.dataframe(collar_data.head())
    
    with col2:
        st.subheader("Fichier des surveys")
        survey_file = st.file_uploader("Sélectionner le fichier survey (CSV, Excel)", type=["csv", "xlsx", "xls"], key="survey_uploader")
        if survey_file is not None:
            survey_data = load_file(survey_file, 'survey')
            if survey_data is not None:
                st.session_state.files['survey'] = survey_data
                st.dataframe(survey_data.head())
    
    st.subheader("Autres fichiers de données")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        assay_file = st.file_uploader("Fichier d'analyses (assays)", type=["csv", "xlsx", "xls"], key="assay_uploader")
        if assay_file is not None:
            assay_data = load_file(assay_file, 'assays')
            if assay_data is not None:
                st.session_state.files['assays'] = assay_data
                st.dataframe(assay_data.head())
    
    with col2:
        litho_file = st.file_uploader("Fichier de lithologie", type=["csv", "xlsx", "xls"], key="litho_uploader")
        if litho_file is not None:
            litho_data = load_file(litho_file, 'litho')
            if litho_data is not None:
                st.session_state.files['litho'] = litho_data
                st.dataframe(litho_data.head())
    
    with col3:
        density_file = st.file_uploader("Fichier de densité", type=["csv", "xlsx", "xls"], key="density_uploader")
        if density_file is not None:
            density_data = load_file(density_file, 'density')
            if density_data is not None:
                st.session_state.files['density'] = density_data
                st.dataframe(density_data.head())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        oxidation_file = st.file_uploader("Fichier d'oxydation", type=["csv", "xlsx", "xls"], key="oxidation_uploader")
        if oxidation_file is not None:
            oxidation_data = load_file(oxidation_file, 'oxidation')
            if oxidation_data is not None:
                st.session_state.files['oxidation'] = oxidation_data
                st.dataframe(oxidation_data.head())
    
    with col2:
        geomet_file = st.file_uploader("Fichier de géométallurgie", type=["csv", "xlsx", "xls"], key="geomet_uploader")
        if geomet_file is not None:
            geomet_data = load_file(geomet_file, 'geometallurgy')
            if geomet_data is not None:
                st.session_state.files['geometallurgy'] = geomet_data
                st.dataframe(geomet_data.head())

with tab2:
    st.header("Validation des données de forage")
    
    if 'collars' not in st.session_state.files:
        st.warning("Veuillez d'abord importer le fichier des collars dans l'onglet 'Import de données'.")
    else:
        if st.button("Lancer la validation des données"):
            with st.spinner("Validation en cours..."):
                # Réinitialiser les résultats
                st.session_state.validation_results = {}
                
                collar_data = st.session_state.files['collars']
                
                # Pour chaque fichier secondaire, effectuer les validations
                for file_type, file_data in st.session_state.files.items():
                    if file_type != 'collars':
                        # Vérifier les forages manquants
                        missing_holes = check_missing_holes(collar_data, file_data, file_type)
                        st.session_state.validation_results[f'missing_holes_{file_type}'] = missing_holes
                        
                        # Vérifier les intervalles de profondeur
                        if file_type in ['assays', 'litho', 'density', 'oxidation', 'geometallurgy']:
                            depth_issues = check_interval_depths(collar_data, file_data, file_type)
                            st.session_state.validation_results[f'depth_issues_{file_type}'] = depth_issues
                
                # Vérifier les doublons dans tous les fichiers
                for file_type, file_data in st.session_state.files.items():
                    duplicates = check_duplicates(file_data, file_type)
                    st.session_state.validation_results[f'duplicates_{file_type}'] = duplicates
                
                st.success("Validation terminée avec succès!")
        
        if st.session_state.validation_results:
            st.header("Résultats de la validation")
            
            # Afficher les problèmes de forages manquants
            st.subheader("Forages manquants")
            
            for result_name, result_data in st.session_state.validation_results.items():
                if 'missing_holes' in result_name and isinstance(result_data, dict):
                    file_type = result_name.split('_')[-1]
                    
                    missing_in_other = result_data.get('missing_in_other', [])
                    extra_in_other = result_data.get('extra_in_other', [])
                    
                    if missing_in_other:
                        st.warning(f"**{len(missing_in_other)}** forages présents dans collars mais absents dans {file_type}")
                        if len(missing_in_other) <= 10:
                            st.write(", ".join(missing_in_other))
                        else:
                            st.write(", ".join(missing_in_other[:10]) + f"... et {len(missing_in_other) - 10} autres")
                    
                    if extra_in_other:
                        st.error(f"**{len(extra_in_other)}** forages présents dans {file_type} mais absents dans collars")
                        if len(extra_in_other) <= 10:
                            st.write(", ".join(extra_in_other))
                        else:
                            st.write(", ".join(extra_in_other[:10]) + f"... et {len(extra_in_other) - 10} autres")
            
            # Afficher les problèmes de profondeur
            st.subheader("Problèmes de profondeur d'intervalles")
            
            depth_issues_exist = False
            for result_name, result_data in st.session_state.validation_results.items():
                if 'depth_issues' in result_name and isinstance(result_data, pd.DataFrame) and not result_data.empty:
                    depth_issues_exist = True
                    file_type = result_name.split('_')[-1]
                    st.error(f"Problèmes de profondeur détectés dans {file_type}: {len(result_data)} intervalles")
                    st.dataframe(result_data)
            
            if not depth_issues_exist:
                st.success("Aucun problème de profondeur détecté dans les intervalles.")
            
            # Afficher les doublons
            st.subheader("Enregistrements en double")
            
            duplicates_exist = False
            for result_name, result_data in st.session_state.validation_results.items():
                if 'duplicates' in result_name and isinstance(result_data, pd.DataFrame) and not result_data.empty:
                    duplicates_exist = True
                    file_type = result_name.split('_')[-1]
                    st.error(f"Doublons détectés dans {file_type}: {len(result_data)} enregistrements")
                    st.dataframe(result_data)
            
            if not duplicates_exist:
                st.success("Aucun doublon détecté dans les fichiers.")
            
            # Bouton pour exporter les résultats
            if st.button("Exporter les résultats de validation"):
                excel_data = export_to_excel(st.session_state.validation_results)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"validation_resultats_{timestamp}.xlsx"
                st.markdown(get_download_link(excel_data, filename), unsafe_allow_html=True)

with tab3:
    st.header("Validation des données de composites")
    
    st.subheader("Import des composites")
    composite_file = st.file_uploader("Sélectionner le fichier de composites (CSV, Excel)", type=["csv", "xlsx", "xls"], key="composite_uploader")
    
    if composite_file is not None:
        composite_data = load_file(composite_file, 'composites')
        if composite_data is not None:
            st.session_state.composite_data = composite_data
            st.dataframe(composite_data.head())
            
            # Réinitialiser le mapping de colonnes
            st.session_state.column_mapping = {}
    
    if st.session_state.composite_data is not None:
        # Interface pour le mapping des coordonnées
        st.subheader("Mapping des coordonnées")
        
        with st.expander("Configurer le mapping des coordonnées", expanded=True):
            st.markdown('<div class="mapping-box">', unsafe_allow_html=True)
            st.markdown("""
            Si vos données de composite utilisent des noms de colonnes différents pour les coordonnées,
            vous pouvez spécifier ici quelles colonnes contiennent les coordonnées X, Y et Z.
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                available_columns = list(st.session_state.composite_data.columns)
                x_column = st.selectbox(
                    "Sélectionner la colonne pour coordonnée X:",
                    options=[""] + available_columns,
                    index=available_columns.index('x') if 'x' in available_columns else 0,
                    key="x_column"
                )
            
            with col2:
                y_column = st.selectbox(
                    "Sélectionner la colonne pour coordonnée Y:",
                    options=[""] + available_columns,
                    index=available_columns.index('y') if 'y' in available_columns else 0,
                    key="y_column"
                )
            
            with col3:
                z_column = st.selectbox(
                    "Sélectionner la colonne pour coordonnée Z:",
                    options=[""] + available_columns,
                    index=available_columns.index('z') if 'z' in available_columns else 0,
                    key="z_column"
                )
            
            # Bouton pour appliquer le mapping
            if st.button("Appliquer le mapping de coordonnées"):
                mapping = {}
                if x_column:
                    mapping['x'] = x_column
                if y_column:
                    mapping['y'] = y_column
                if z_column:
                    mapping['z'] = z_column
                
                if len(mapping) == 3:
                    st.session_state.column_mapping = mapping
                    st.success("Mapping de coordonnées appliqué!")
                else:
                    st.error("Veuillez sélectionner toutes les colonnes de coordonnées (X, Y et Z).")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Paramètres de validation
        st.subheader("Paramètres de validation des composites")
        
        distance_threshold = st.slider("Seuil de distance pour échantillons proches (m)", 0.01, 5.0, 0.5, 0.01)
        
        # Afficher un résumé du mapping actuel
        if st.session_state.column_mapping:
            st.info(f"""
            Mapping actuel des coordonnées:
            - X ➔ {st.session_state.column_mapping['x']}
            - Y ➔ {st.session_state.column_mapping['y']}
            - Z ➔ {st.session_state.column_mapping['z']}
            """)
        
        if st.button("Vérifier les composites proches"):
            with st.spinner("Analyse en cours..."):
                # Utiliser le mapping de colonnes s'il existe
                result = check_nearby_composites(
                    st.session_state.composite_data,
                    distance_threshold,
                    st.session_state.column_mapping if st.session_state.column_mapping else None
                )
                
                if result is None:
                    st.error("Erreur inattendue lors de l'analyse des composites proches.")
                elif result['status'] == 'error':
                    st.error(f"Erreur: {result.get('message', 'Une erreur est survenue pendant l\'analyse')}")
                elif result['status'] == 'empty':
                    st.success(f"Aucune paire d'échantillons composites n'a été trouvée à moins de {distance_threshold} mètres l'un de l'autre.")
                elif result['status'] == 'success':
                    close_points = result['points']
                    st.warning(f"Détection de {result['count']} paires d'échantillons composites proches")
                    
                    # Créer un DataFrame pour afficher les résultats
                    close_df = pd.DataFrame([{
                        'holeid_1': cp['point1_holeid'],
                        'holeid_2': cp['point2_holeid'],
                        'distance (m)': cp['distance'],
                        'x1': cp['point1_coords'][0],
                        'y1': cp['point1_coords'][1],
                        'z1': cp['point1_coords'][2],
                        'x2': cp['point2_coords'][0],
                        'y2': cp['point2_coords'][1],
                        'z2': cp['point2_coords'][2]
                    } for cp in close_points])
                    
                    st.dataframe(close_df)
                    
                    # Afficher une visualisation 3D des points proches
                    if st.checkbox("Afficher une visualisation 3D des points proches"):
                        # Déterminer les colonnes à utiliser pour la visualisation
                        x_col = st.session_state.column_mapping.get('x', 'x') if st.session_state.column_mapping else 'x'
                        y_col = st.session_state.column_mapping.get('y', 'y') if st.session_state.column_mapping else 'y'
                        z_col = st.session_state.column_mapping.get('z', 'z') if st.session_state.column_mapping else 'z'
                        
                        fig = go.Figure()
                        
                        # Ajouter tous les points
                        fig.add_trace(go.Scatter3d(
                            x=st.session_state.composite_data[x_col],
                            y=st.session_state.composite_data[y_col],
                            z=st.session_state.composite_data[z_col],
                            mode='markers',
                            marker=dict(
                                size=3,
                                color='blue',
                                opacity=0.5
                            ),
                            name='Tous les composites'
                        ))
                        
                        # Ajouter les points proches
                        close_indices = set()
                        for cp in close_points:
                            close_indices.add(cp['point1_index'])
                            close_indices.add(cp['point2_index'])
                        
                        close_points_df = st.session_state.composite_data.iloc[list(close_indices)]
                        
                        fig.add_trace(go.Scatter3d(
                            x=close_points_df[x_col],
                            y=close_points_df[y_col],
                            z=close_points_df[z_col],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color='red',
                            ),
                            name='Composites proches'
                        ))
                        
                        fig.update_layout(
                            title='Visualisation 3D des composites',
                            scene=dict(
                                xaxis_title='X',
                                yaxis_title='Y',
                                zaxis_title='Z',
                                aspectmode='data'
                            ),
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            ),
                            margin=dict(l=0, r=0, b=0, t=30)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Permettre l'exportation des résultats
                        if st.button("Exporter les résultats des composites proches"):
                            output = io.BytesIO()
                            close_df.to_excel(output, index=False)
                            output.seek(0)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"composites_proches_{timestamp}.xlsx"
                            st.markdown(get_download_link(output.getvalue(), filename), unsafe_allow_html=True)

# Pied de page
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 14px;">
© 2025 Didier Ouedraogo, P.Geo | Application de validation de données de forage minier
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from catboost import CatBoostRegressor, Pool
import pickle
from itertools import chain
import ast
from functools import reduce
import re
from typing import Dict, List, Optional
import heapq

# ==============================================================================
# KELAS PIPELINE
# ==============================================================================
class ContainerRepairPipeline:
    def __init__(self):
        self.cat_cols = ["IDKONTRAKTOR", "CONTAINER_GRADE", "CONTAINER_SIZE", "CONTAINER_TYPE"]
        self.depo_config = {
            "SBY": {
                "vendors": ['MTCP', 'SPIL'],
                "model_path": "SBY_SPIL_Final.cbm",
                "model_mhr_path": "SBY_model_MHR.cbm",
                "category_map_path": 'sby_category_threshold_maps.pkl'
            },
            "JKT": {
                "vendors": ['MDS', 'SPIL', 'MDSBC', 'MACBC', 'PTMAC', 'MCPNL', 'MCPCONCH'],
                "model_path": "JKT_SPIL_Final.cbm",
                "model_mhr_path": "JKT_model_MHR.cbm",
                "category_map_path": 'jkt_category_threshold_maps.pkl'
            }
        }
        self.validity_map = { "JKT": { 'MDS': ['A'], 'SPIL': ['A', 'B', 'C'], 'MDSBC': ['B', 'C'], 'MACBC': ['B', 'C'], 'PTMAC': ['A'], 'MCPNL': ['A', 'B', 'C'], 'MCPCONCH': ['B', 'C'] }, "SBY": { 'MTCP': ['A', 'B', 'C'], 'SPIL': ['A', 'B', 'C'] } }
        self.models, self.category_maps, self.models_mhr = {}, {}, {}

    def load_models(self):
        for depo, config in self.depo_config.items():
            try:
                model = CatBoostRegressor()
                model.load_model(config["model_path"])
                self.models[depo] = model
            except Exception as e:
                print(f"Warning: Gagal memuat model utama untuk DEPO {depo}. Error: {e}")

            if "model_mhr_path" in config:
                try:
                    model_mhr = CatBoostRegressor()
                    model_mhr.load_model(config["model_mhr_path"])
                    self.models_mhr[depo] = model_mhr
                except Exception as e:
                    print(f"Warning: Gagal memuat model MHR untuk DEPO {depo}. Error: {e}")
            
            try:
                with open(config["category_map_path"], 'rb') as f:
                    self.category_maps[depo] = pickle.load(f)
            except Exception as e:
                print(f"Warning: Gagal memuat map untuk DEPO {depo}. Error: {e}")

    def run_pipeline(self, input_data: pd.DataFrame) -> pd.DataFrame:
        df_processed = self.preprocess_data(input_data)
        df_cat = self.process_categories(df_processed)
        df_expanded = self.expand_quantity(df_cat)
        df_agg = self.aggregate_data(df_expanded)
        one_hot = self.one_hot_encode(df_agg)
        final_data = self.prepare_final_data(df_agg, one_hot)
        return self.predict(final_data)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        required_cols = ['NO_EOR', 'CONTAINER_SIZE', 'CONTAINER_GRADE', 'DAMAGE', 'REPAIRACTION', 'COMPONENT', 'LOCATION', 'QTY']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Kolom wajib tidak ditemukan: {missing_cols}")
        df['CONTAINER_TYPE'] = (df['CONTAINER_SIZE'].astype(str) + df['CONTAINER_GRADE'].astype(str)).astype("category")
        df['LOCATION'] = (df['LOCATION'].astype(str).str.replace(r'(\d+\s*(?:st|nd|rd|th)?)\s*,\s*(\d+\s*(?:st|nd|rd|th)?)', r'\1-\2', flags=re.IGNORECASE, regex=True).str.upper().str.replace(r'\s*-\s*', '-', regex=True).str.replace(r'\.$', '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip())
        df["DAMAGE_COMPONENT"] = df["DAMAGE"].astype(str) + "_" + df["COMPONENT"].astype(str)
        df["DAMAGE_ACTION"] = df["DAMAGE"].astype(str) + "_" + df["REPAIRACTION"].astype(str)
        df["COMPONENT_ACTION"] = df["COMPONENT"].astype(str) + "_" + df["REPAIRACTION"].astype(str)
        df["LOCATION_DAMAGE"] = df["LOCATION"].astype(str) + "_" + df["DAMAGE"].astype(str)
        return df

    def process_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        columns = ['COMPONENT', 'LOCATION', 'DAMAGE', 'REPAIRACTION', 'DAMAGE_ACTION', 'DAMAGE_COMPONENT', 'COMPONENT_ACTION', 'LOCATION_DAMAGE']
        for depo in data["DEPO"].unique():
            if depo not in self.category_maps:
                continue
            mask = data["DEPO"] == depo
            category_map = self.category_maps[depo]
            for col in columns:
                if col in data.columns:
                    valid_categories = category_map.get(col, [])
                    condition = ~data.loc[mask, col].isin(valid_categories)
                    data.loc[mask & condition, col] = 'other'
        return data

    def expand_quantity(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        list_cols = ["COMPONENT", "LOCATION", "DAMAGE", "REPAIRACTION", "DAMAGE_ACTION", "DAMAGE_COMPONENT", "COMPONENT_ACTION", "LOCATION_DAMAGE"]
        df["QTY"] = pd.to_numeric(df["QTY"], errors='coerce').fillna(0).astype("int8")
        for col in list_cols:
            if col in df.columns:
                df[col] = [[val] * qty for val, qty in zip(df[col], df["QTY"])]
        return df

    def aggregate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        first_cols = ["NO_EOR", "CONTAINER_GRADE", "CONTAINER_SIZE", "CONTAINER_TYPE", "DEPO"]
        list_cols = ["COMPONENT", "LOCATION", "DAMAGE", "REPAIRACTION", "DAMAGE_ACTION", "DAMAGE_COMPONENT", "COMPONENT_ACTION", "LOCATION_DAMAGE"]
        
        agg_dict = {col: "first" for col in first_cols if col != "NO_EOR"}
        for col in list_cols:
            if col in data.columns:
                agg_dict[col] = lambda series: list(chain.from_iterable(v if isinstance(v, list) else [v] for v in series.dropna()))
        
        return data.groupby("NO_EOR", as_index=False).agg(agg_dict)

    def one_hot_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        list_cols = ["COMPONENT", "LOCATION", "DAMAGE", "REPAIRACTION", "DAMAGE_ACTION", "DAMAGE_COMPONENT", "COMPONENT_ACTION", "LOCATION_DAMAGE"]
        results = {}
        
        def safe_parse_list(val):
            if isinstance(val, str):
                try: return ast.literal_eval(val)
                except (ValueError, SyntaxError): return []
            return val if isinstance(val, list) else []

        for column in list_cols:
            if column not in data.columns: continue
            
            df_col = data[["NO_EOR", column]].copy()
            df_col[column] = df_col[column].apply(safe_parse_list)
            df_exploded = df_col.explode(column).dropna(subset=[column])
            
            if df_exploded.empty: continue
            
            dummies = pd.get_dummies(df_exploded[column], prefix=column)
            df_encoded = pd.concat([df_exploded[["NO_EOR"]], dummies], axis=1)
            results[column] = df_encoded.groupby("NO_EOR", as_index=False).sum()
        
        if not results:
            return pd.DataFrame({'NO_EOR': data['NO_EOR']})
        
        return reduce(lambda left, right: pd.merge(left, right, on="NO_EOR", how="outer"), results.values())

    def prepare_final_data(self, data: pd.DataFrame, one_hot: pd.DataFrame) -> pd.DataFrame:
        df_agg = data.drop(columns=[col for col in ["COMPONENT", "LOCATION", "DAMAGE", "REPAIRACTION", "DAMAGE_ACTION", "DAMAGE_COMPONENT", "COMPONENT_ACTION", "LOCATION_DAMAGE"] if col in data.columns])
        df_merged = pd.merge(df_agg, one_hot, on="NO_EOR", how="left")
        
        one_hot_cols = [col for col in one_hot.columns if col != 'NO_EOR']
        for col in one_hot_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(0)
                
        return df_merged

    def predict(self, final_data: pd.DataFrame) -> pd.DataFrame:
        if "DEPO" not in final_data.columns:
            raise ValueError("DEPO column not found")
        
        all_results = []
        for depo in final_data["DEPO"].unique():
            model = self.models.get(depo)
            model_mhr = self.models_mhr.get(depo)
            if not model:
                continue

            df_depo = final_data[final_data["DEPO"] == depo].copy()
            base_df_for_pred = df_depo.drop(columns=["NO_EOR", "DEPO"], errors='ignore')

            for vendor in self.depo_config[depo]["vendors"]:
                expected_features = model.feature_names_
                df_copy_pred = base_df_for_pred.copy()
                
                missing_cols = set(expected_features) - set(df_copy_pred.columns)
                for col in missing_cols: df_copy_pred[col] = 0
                df_copy_pred = df_copy_pred[expected_features]
                df_copy_pred["IDKONTRAKTOR"] = vendor
                
                cat_features_pred = [c for c in self.cat_cols if c in df_copy_pred.columns]
                for col in cat_features_pred: df_copy_pred[col] = df_copy_pred[col].astype(str)
                
                pool_pred = Pool(data=df_copy_pred, cat_features=cat_features_pred)
                preds = model.predict(pool_pred)
                
                valid_grades = self.validity_map.get(depo, {}).get(vendor, [])
                mask_invalid = ~df_depo['CONTAINER_GRADE'].isin(valid_grades)
                preds[mask_invalid] = np.nan
                df_depo[f"PREDIKSI_{vendor}"] = preds

                preds_mhr = np.full_like(preds, np.nan)
                if model_mhr:
                    mhr_features = model_mhr.feature_names_
                    df_copy_mhr = base_df_for_pred.copy()
                    
                    missing_mhr_cols = set(mhr_features) - set(df_copy_mhr.columns)
                    for col in missing_mhr_cols: df_copy_mhr[col] = 0
                    df_copy_mhr = df_copy_mhr[mhr_features]
                    df_copy_mhr["IDKONTRAKTOR"] = vendor
                    
                    cat_features_mhr = [c for c in self.cat_cols if c in df_copy_mhr.columns]
                    for col in cat_features_mhr: df_copy_mhr[col] = df_copy_mhr[col].astype(str)

                    pool_mhr = Pool(data=df_copy_mhr, cat_features=cat_features_mhr)
                    preds_mhr_raw = model_mhr.predict(pool_mhr)
                    preds_mhr_raw[mask_invalid] = np.nan
                    preds_mhr = preds_mhr_raw
                
                df_depo[f"MHR_{vendor}"] = preds_mhr

                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.divide(preds, preds_mhr)
                df_depo[f"PREDIKSI/MHR_{vendor}"] = ratio

            all_results.append(df_depo)
            
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# ==============================================================================
# UI STREAMLIT DAN LOGIKA APLIKASI
# ==============================================================================
@st.cache_resource
def get_pipeline():
    """
    Loads the pipeline and its models once and caches it.
    This prevents reloading on every script rerun.
    """
    pipeline = ContainerRepairPipeline()
    pipeline.load_models()  # Models are loaded here, just once.
    return pipeline

st.set_page_config(page_title="Container Repair Prediction", layout="wide")
st.title("Dashboard Alokasi Perbaikan Kontainer")

# Get the cached pipeline instance.
pipeline = get_pipeline()

with st.sidebar:
    st.header("⚙️ Parameter Global")
    depo_option = st.selectbox("Pilih DEPO", ["JKT", "SBY"], key="global_depo")
    
tab_manual, tab_bulk = st.tabs(["Input Manual", "Input CSV"])

# ==============================================================================
# TAB 1: CEK HARGA MANUAL (EFFICIENT VERSION)
# ==============================================================================
with tab_manual:
    st.header("Estimasi Biaya Perbaikan Berdasarkan Jenis Kerusakan")
    st.info("Masukkan detail perbaikan untuk satu kontainer untuk melihat perbandingan prediksi biaya, MHR, dan rasio antar vendor.")

    st.subheader("📝 Masukkan Detail Kerusakan")
    num_entries = st.number_input("Jumlah Item Kerusakan", min_value=1, max_value=30, value=3, help="Tentukan berapa banyak baris kerusakan yang akan Anda masukkan.")

    @st.cache_data
    def load_unique_values():
        try:
            return pd.read_csv("list_unique.csv")
        except FileNotFoundError:
            st.error("Error: 'list_unique.csv' tidak ditemukan. Menggunakan daftar kosong.")
            return pd.DataFrame({'DAMAGE': [], 'REPAIRACTION': [], 'LOCATION': [], 'COMPONENT': []})

    unique_list_df = load_unique_values()
    
    DAMAGE_OPTIONS = ["- Pilih Tipe Kerusakan -"] + sorted(unique_list_df["DAMAGE"].dropna().unique().tolist())
    REPAIR_OPTIONS = ["- Pilih Tindakan -"] + sorted(unique_list_df["REPAIRACTION"].dropna().unique().tolist())
    LOCATION_OPTIONS = ["- Pilih Lokasi -"] + sorted(unique_list_df["LOCATION"].dropna().unique().tolist())
    COMPONENT_OPTIONS = ["- Pilih Komponen -"] + sorted(unique_list_df["COMPONENT"].dropna().unique().tolist())

    with st.form("manual_entry_form"):
        container_grade = st.selectbox("Kontainer Grade", ['A', 'B', 'C'], help="Grade kontainer.", key="manual_grade")
        container_size = st.selectbox("Ukuran Kontainer", ['20', '40'], help="Ukuran kontainer (20ft atau 40ft).", key="manual_size")
        damage_data = {'damage': [], 'repair_action': [], 'location': [], 'component': [], 'qty': []}
        
        cols_header = st.columns(5)
        cols_header[0].markdown("**Tipe Kerusakan**")
        cols_header[1].markdown("**Tindakan Perbaikan**")
        cols_header[2].markdown("**Lokasi**")
        cols_header[3].markdown("**Komponen**")
        cols_header[4].markdown("**Kuantitas**")
        
        for i in range(num_entries):
            cols = st.columns(5)
            damage_data['damage'].append(cols[0].selectbox(f"Damage_{i}", DAMAGE_OPTIONS, key=f"damage_{i}", label_visibility="collapsed"))
            damage_data['repair_action'].append(cols[1].selectbox(f"Repair_{i}", REPAIR_OPTIONS, key=f"repair_{i}", label_visibility="collapsed"))
            damage_data['location'].append(cols[2].selectbox(f"Location_{i}", LOCATION_OPTIONS, key=f"location_{i}", label_visibility="collapsed"))
            damage_data['component'].append(cols[3].selectbox(f"Component_{i}", COMPONENT_OPTIONS, key=f"component_{i}", label_visibility="collapsed"))
            damage_data['qty'].append(cols[4].number_input(f"Qty_{i}", min_value=1, value=1, key=f"qty_{i}", label_visibility="collapsed"))

        submitted = st.form_submit_button("Cek Estimasi")

    if submitted:
        fields_to_validate = [damage_data['damage'], damage_data['repair_action'], damage_data['location'], damage_data['component']]
        if any(opt.startswith("- Pilih") for opt in chain.from_iterable(fields_to_validate)):
            st.warning("Mohon pastikan semua detail kerusakan (Tipe, Tindakan, Lokasi, Komponen) telah dipilih dari dropdown.")
        else:
            with st.spinner("Loading..."):
                manual_input_rows = [{"NO_EOR": "MANUAL_CHECK", "CONTAINER_SIZE": container_size, "CONTAINER_GRADE": container_grade, "DAMAGE": damage_data['damage'][i], "REPAIRACTION": damage_data['repair_action'][i], "COMPONENT": damage_data['component'][i], "LOCATION": damage_data['location'][i], "QTY": damage_data['qty'][i], "DEPO": depo_option} for i in range(num_entries)]
                manual_df = pd.DataFrame(manual_input_rows)
                
                try:
                    prediction_result = pipeline.run_pipeline(manual_df)
                    
                    if not prediction_result.empty:
                        result_row = prediction_result.iloc[0]
                        st.subheader(f"Hasil Estimasi untuk DEPO {depo_option}")
                        display_data = [{"Vendor": vendor, "Prediksi Biaya": result_row.get(f"PREDIKSI_{vendor}", np.nan), "Estimasi MHR": result_row.get(f"MHR_{vendor}", np.nan), "Rasio Biaya/MHR": result_row.get(f"PREDIKSI/MHR_{vendor}", np.nan)} for vendor in pipeline.depo_config.get(depo_option, {}).get("vendors", [])]
                        
                        price_df = pd.DataFrame(display_data).dropna(subset=['Prediksi Biaya', 'Estimasi MHR'], how='all').sort_values(by="Prediksi Biaya", na_position='last')
                        if not price_df.empty:
                            st.dataframe(price_df.style.format({'Prediksi Biaya': 'Rp {:,.0f}', 'Estimasi MHR': '{:,.2f}', 'Rasio Biaya/MHR': 'Rp {:,.0f}/jam'}, na_rep='-'), use_container_width=True)
                        else:
                            st.error("Tidak ada prediksi yang valid untuk kombinasi Grade dan DEPO yang dipilih.")
                    else:
                        st.error("Gagal mendapatkan hasil prediksi.")
                except Exception as e:
                    st.error(f"Terjadi error saat prediksi manual: {e}")

# ==============================================================================
# TAB 2: ALOKASI OPTIMAL (BULK) - LOGIKA BARU
# ==============================================================================
with tab_bulk:
    st.header("Alokasi Optimal untuk Perbaikan Kontainer")

    with st.expander("Opsi & Upload File", expanded=True):
        uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"], key="bulk_upload_spil")

        st.markdown("##### **Kapasitas SPIL**")
        
        col_toggle1, col_toggle2 = st.columns(2)
        use_container_filter = col_toggle1.toggle("Gunakan Filter Kapasitas Kontainer", value=True, key="toggle_container")
        use_mhr_filter = col_toggle2.toggle("Gunakan Filter Kapasitas MHR", value=True, key="toggle_mhr")
        
        col1_spil, col2_spil = st.columns(2)
        spil_container_capacity = col1_spil.number_input(
            f"Kapasitas Kontainer SPIL", 
            min_value=0, 
            value=100, 
            key=f"today_container_spil",
            disabled=not use_container_filter
        )
        spil_mhr_capacity = col2_spil.number_input(
            f"Kapasitas MHR SPIL", 
            min_value=0, 
            value=5000, 
            key=f"today_mhr_spil", 
            format="%d",
            disabled=not use_mhr_filter
        )

        st.markdown("---")
        st.markdown("##### **Penanganan Sisa Pekerjaan (Opsional)**")

        use_waiting_list = st.checkbox("Waiting List SPIL", key="use_waiting_list")
        tomorrow_capacities_input = {}
        if use_waiting_list:
            col1_wl, col2_wl = st.columns(2)
            tomorrow_container_capacity = col1_wl.number_input(
                "Kapasitas Kontainer SPIL Besok", 
                min_value=0, 
                value=50, 
                key="tomorrow_container_spil",
                disabled=not use_container_filter
            )
            tomorrow_mhr_capacity = col2_wl.number_input(
                "Kapasitas MHR SPIL Besok", 
                min_value=0, 
                value=2500, 
                key="tomorrow_mhr_spil", 
                format="%d",
                disabled=not use_mhr_filter
            )
            tomorrow_capacities_input = {"kontainer": tomorrow_container_capacity, "mhr": tomorrow_mhr_capacity}

        use_other_vendors = st.checkbox("Vendor Lain", key="use_other_vendors")
        other_vendor_capacities_input = {}
        if use_other_vendors:
            other_vendors = [v for v in pipeline.depo_config.get(depo_option, {}).get("vendors", []) if v != 'SPIL']
            for vendor in other_vendors:
                col1_other, col2_other = st.columns(2)
                container_capacity = col1_other.number_input(
                    f"Kapasitas Kontainer {vendor}", 
                    min_value=0, 
                    value=100, 
                    key=f"other_container_{vendor}",
                    disabled=not use_container_filter
                )
                mhr_capacity = col2_other.number_input(
                    f"Kapasitas MHR {vendor}", 
                    min_value=0, 
                    value=5000, 
                    key=f"other_mhr_{vendor}", 
                    format="%d",
                    disabled=not use_mhr_filter
                )
                other_vendor_capacities_input[vendor] = {"kontainer": container_capacity, "mhr": mhr_capacity}

        run_bulk_button = st.button("Cek Alokasi", type="primary", key="spil_run")

    @st.cache_data
    def run_spil_centric_allocation(_pipeline, uploaded_file_content, depo_option, spil_today_cap, spil_tomorrow_cap, other_vendor_caps, use_wl, use_ov, use_container_filter, use_mhr_filter, version="3.4"):
        data = pd.read_csv(StringIO(uploaded_file_content))
        data["DEPO"] = depo_option
        
        raw_results = _pipeline.run_pipeline(data)
        if 'PREDIKSI_SPIL' not in raw_results.columns:
            st.error("Prediksi untuk SPIL tidak tersedia. Alokasi tidak dapat dilanjutkan.")
            return pd.DataFrame()

        other_vendor_preds = [c for c in raw_results.columns if c.startswith('PREDIKSI_') and 'SPIL' not in c and not c.startswith('PREDIKSI/MHR_')]
        raw_results['Prediksi_Biaya_Lain'] = raw_results[other_vendor_preds].min(axis=1)
        raw_results['Selisih_Prediksi_Biaya'] = raw_results['Prediksi_Biaya_Lain'] - raw_results['PREDIKSI_SPIL']
        
        spil_candidates = raw_results.sort_values(by='Selisih_Prediksi_Biaya', ascending=False)
        allocations = {}
        
        spil_container_cap = spil_today_cap['kontainer'] if use_container_filter else float('inf')
        spil_mhr_cap = spil_today_cap['mhr'] if use_mhr_filter else float('inf')
        unallocated_eors = []
        
        for idx, row in spil_candidates.iterrows():
            eor = row['NO_EOR']
            mhr_needed = row.get('MHR_SPIL', 0)
            if pd.isna(mhr_needed): mhr_needed = 0

            if (not use_container_filter or spil_container_cap > 0) and (not use_mhr_filter or spil_mhr_cap >= mhr_needed):
                allocations[eor] = {'ALOKASI': 'SPIL', 'HARGA_FINAL': row['PREDIKSI_SPIL']}
                if use_container_filter:
                    spil_container_cap -= 1
                if use_mhr_filter:
                    spil_mhr_cap -= mhr_needed
            else:
                unallocated_eors.append(eor)
        
        overflow_df = spil_candidates[spil_candidates['NO_EOR'].isin(unallocated_eors)].copy()
        
        if use_wl:
            waiting_list_candidates = overflow_df.sort_values(by='Selisih_Prediksi_Biaya', ascending=False)
            spil_tomorrow_container_cap = spil_tomorrow_cap.get('kontainer', 0) if use_container_filter else float('inf')
            spil_tomorrow_mhr_cap = spil_tomorrow_cap.get('mhr', 0) if use_mhr_filter else float('inf')
            remaining_after_wl = []
            
            for idx, row in waiting_list_candidates.iterrows():
                eor = row['NO_EOR']
                mhr_needed = row.get('MHR_SPIL', 0)
                if pd.isna(mhr_needed): mhr_needed = 0
                if (not use_container_filter or spil_tomorrow_container_cap > 0) and (not use_mhr_filter or spil_tomorrow_mhr_cap >= mhr_needed):
                    allocations[eor] = {'ALOKASI': 'Waiting List SPIL', 'HARGA_FINAL': row['PREDIKSI_SPIL']}
                    if use_container_filter:
                        spil_tomorrow_container_cap -= 1
                    if use_mhr_filter:
                        spil_tomorrow_mhr_cap -= mhr_needed
                else:
                    remaining_after_wl.append(eor)
            overflow_df = overflow_df[overflow_df['NO_EOR'].isin(remaining_after_wl)].copy()

        if use_ov:
            other_vendor_candidates = overflow_df.sort_values(by='Selisih_Prediksi_Biaya', ascending=True)
            for idx, row in other_vendor_candidates.iterrows():
                eor = row['NO_EOR']
                allocated = False
                cheapest_options = row[other_vendor_preds].dropna().sort_values()
                for vendor_price_val in cheapest_options.items():
                    vendor_name = vendor_price_val[0].replace('PREDIKSI_', '')
                    mhr_needed = row.get(f'MHR_{vendor_name}', 0)
                    if pd.isna(mhr_needed): mhr_needed = 0

                    container_cap = other_vendor_caps.get(vendor_name, {}).get('kontainer', 0) if use_container_filter else float('inf')
                    mhr_cap = other_vendor_caps.get(vendor_name, {}).get('mhr', 0) if use_mhr_filter else float('inf')

                    if (not use_container_filter or container_cap > 0) and (not use_mhr_filter or mhr_cap >= mhr_needed):
                        allocations[eor] = {'ALOKASI': f'{vendor_name}', 'HARGA_FINAL': vendor_price_val[1]}
                        if use_container_filter:
                            other_vendor_caps[vendor_name]['kontainer'] -= 1
                        if use_mhr_filter:
                            other_vendor_caps[vendor_name]['mhr'] -= mhr_needed
                        allocated = True
                        break
                if not allocated:
                    allocations[eor] = {'ALOKASI': 'Tidak Terhandle', 'HARGA_FINAL': np.nan}
        else:
            for eor in overflow_df['NO_EOR']:
                allocations[eor] = {'ALOKASI': 'Tidak Terhandle', 'HARGA_FINAL': np.nan}

        allocations_df = pd.DataFrame.from_dict(allocations, orient='index')
        final_df = raw_results.set_index('NO_EOR').join(allocations_df).reset_index()
        return final_df

    if run_bulk_button and uploaded_file is not None:
        try:
            uploaded_file_content = uploaded_file.getvalue().decode('utf-8')
            
            spil_today_caps = {"kontainer": spil_container_capacity, "mhr": spil_mhr_capacity}

            with st.spinner(f'Loading...'):
                final_results = run_spil_centric_allocation(
                    pipeline, uploaded_file_content, depo_option, 
                    spil_today_caps, tomorrow_capacities_input, 
                    other_vendor_capacities_input, use_waiting_list, use_other_vendors,
                    use_container_filter, use_mhr_filter,
                    version="3.4"
                )
            
            if not final_results.empty:
                st.success("✅ Alokasi berhasil diselesaikan!")
                
                def get_final_mhr(row):
                    if pd.isna(row['ALOKASI']) or 'Tidak Terhandle' in row['ALOKASI']: return np.nan
                    if 'SPIL' in row['ALOKASI']: vendor = 'SPIL'
                    else: vendor = row['ALOKASI'].replace('', '')
                    return row.get(f"MHR_{vendor}", np.nan)
                final_results['MHR'] = final_results.apply(get_final_mhr, axis=1)

                st.markdown("---")
                st.subheader("Ringkasan Hasil Alokasi")
                
                vendor_stats = final_results.groupby('ALOKASI').agg(
                    Jumlah_Kontainer=('NO_EOR', 'nunique'),
                    Total_Prediksi_Biaya=('HARGA_FINAL', 'sum'),
                    Total_MHR=('MHR', 'sum')
            ).reset_index().rename(columns={'ALOKASI': 'STATUS', 'Total_Prediksi_Biaya': 'Total Biaya'})
                
                st.dataframe(vendor_stats.style.format({
                    'Total Biaya': 'Rp {:,.0f}', 
                    'Total_MHR': '{:,.2f}'
                }), use_container_width=True)

                st.markdown("---")
                st.subheader("Detail Hasil Alokasi")
                
                display_cols = ['NO_EOR', 'CONTAINER_TYPE', 'ALOKASI', 'HARGA_FINAL', 'MHR', 'Selisih_Prediksi_Biaya', 'PREDIKSI_SPIL', 'Prediksi_Biaya_Lain']
                
                display_cols_exist = [col for col in display_cols if col in final_results.columns]
                display_df = final_results[display_cols_exist].sort_values(by='Selisih_Prediksi_Biaya', ascending=False)
                
                st.dataframe(
                    display_df.rename(columns={
                        'HARGA_FINAL': 'Prediksi Biaya',
                        'PREDIKSI_SPIL': 'Prediksi Biaya SPIL',
                        'Selisih_Prediksi_Biaya': 'Potensial Keuntungan',
                        'CONTAINER_TYPE': 'Tipe Kontainer',
                        'ALOKASI': 'Alokasi',
                        'NO_EOR': 'No EOR',
                        'Prediksi_Biaya_Lain': 'Prediksi Biaya Lain',
                        'MHR': 'Estimasi MHR'
                    }).style.format({
                        'Prediksi Biaya': 'Rp {:,.0f}',
                        'Estimasi MHR': '{:,.2f}',
                        'Potensial Keuntungan': 'Rp {:,.0f}',
                        'Prediksi Biaya SPIL': 'Rp {:,.0f}',
                        'Prediksi Biaya Lain': 'Rp {:,.0f}',
                    }, na_rep='-'),
                    height=600,
                    use_container_width=True
                )
                
                csv_final = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Hasil Alokasi", data=csv_final, file_name=f"hasil_alokasi_spil_{depo_option}.csv", mime="text/csv")
                
                with st.expander("Lihat Tabel Alokasi Lengkap", expanded=False):
                    st.caption("Tabel ini menampilkan hasil alokasi final dan semua detail prediksi. Kolom prediksi biaya dan MHR yang terpilih diberi highlight kuning.")

                    def highlight_final_choice(row):
                        highlight_color = 'background-color: #fff8c4;'
                        styles = [''] * len(row)
                        alokasi = row.get('ALOKASI')
                        if pd.isna(alokasi) or 'Tidak Terhandle' in alokasi:
                            return styles
                        
                        vendor = alokasi.replace('', '').replace('Waiting List ', '').strip()
                        pred_col = f'PREDIKSI_{vendor}'
                        mhr_col = f'MHR_{vendor}'
                        
                        try:
                            if pred_col in row.index:
                                pred_idx = row.index.get_loc(pred_col)
                                styles[pred_idx] = highlight_color
                            if mhr_col in row.index:
                                mhr_idx = row.index.get_loc(mhr_col)
                                styles[mhr_idx] = highlight_color
                        except KeyError:
                            pass
                        return styles

                    base_info_cols = ['NO_EOR', 'CONTAINER_TYPE', 'ALOKASI', 'HARGA_FINAL', 'MHR', 'Selisih_Prediksi_Biaya']
                    pred_cols_all = sorted([col for col in final_results.columns if col.startswith("PREDIKSI_") and not col.startswith("PREDIKSI/MHR_")])
                    mhr_cols_all = sorted([col for col in final_results.columns if col.startswith("MHR_") and col != 'MHR'])
                    
                    comprehensive_cols = base_info_cols + pred_cols_all + mhr_cols_all
                    comprehensive_cols_exist = [col for col in comprehensive_cols if col in final_results.columns]
                    
                    detail_df = final_results[comprehensive_cols_exist].sort_values(by='Selisih_Prediksi_Biaya', ascending=False).copy()
                    
                    format_dict_full = {
                        'Prediksi Biaya Final': 'Rp {:,.0f}',
                        'MHR': '{:,.2f}',
                        'Selisih_Prediksi_Biaya': 'Rp {:,.0f}'
                    }
                    for col in pred_cols_all: format_dict_full[col] = 'Rp {:,.0f}'
                    for col in mhr_cols_all: format_dict_full[col] = '{:,.2f}'
                    
                    st.dataframe(detail_df.rename(columns={'HARGA_FINAL': 'Prediksi Biaya Final'}).style.apply(highlight_final_choice, axis=1).format(format_dict_full, na_rep='-'), height=600, use_container_width=True)
                    
                    csv_pred_detail = detail_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Tabel Lengkap",
                        data=csv_pred_detail,
                        file_name=f"prediksi_super_lengkap_{depo_option}.csv",
                        mime="text/csv",
                        key="download_super_lengkap"
                    )

        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")
            st.exception(e)

    elif run_bulk_button:
        st.warning("Mohon unggah file CSV terlebih dahulu.")
    else:
        st.info("Silakan unggah file CSV dan klik 'Cek Alokasi' untuk memulai.")
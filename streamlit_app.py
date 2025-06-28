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
# KELAS PIPELINE (TIDAK ADA PERUBAHAN)
# ==============================================================================
class ContainerRepairPipeline:
    def __init__(self):
        self.cat_cols = ["IDKONTRAKTOR", "CONTAINER_GRADE", "CONTAINER_SIZE", "CONTAINER_TYPE"]
        self.depo_config = { "SBY": { "vendors": ['MTCP', 'SPIL'], "model_path": "SBY_model_23_Juni.cbm", "category_map_path": 'sby_category_threshold_maps.pkl' }, "JKT": { "vendors": ['MDS', 'SPIL', 'MDSBC', 'MACBC', 'PTMAC', 'MCPNL', 'MCPCONCH'], "model_path": "JKT_model_23_Juni.cbm", "category_map_path": 'jkt_category_threshold_maps.pkl' } }
        self.validity_map = { "JKT": { 'MDS': ['A'], 'SPIL': ['A', 'B', 'C'], 'MDSBC': ['B', 'C'], 'MACBC': ['B', 'C'], 'PTMAC': ['A'], 'MCPNL': ['A', 'B', 'C'], 'MCPCONCH': ['B', 'C'] }, "SBY": { 'MTCP': ['A', 'B', 'C'], 'SPIL': ['A', 'B', 'C'] } }
        self.models, self.category_maps = {}, {}
    def load_models(self):
        for depo, config in self.depo_config.items():
            model = CatBoostRegressor();
            try: model.load_model(config["model_path"]); self.models[depo] = model;
            except Exception as e: print(f"Warning: Gagal memuat model untuk DEPO {depo}. Error: {e}")
            try:
                with open(config["category_map_path"], 'rb') as f: self.category_maps[depo] = pickle.load(f)
            except Exception as e: print(f"Warning: Gagal memuat map untuk DEPO {depo}. Error: {e}")
    def run_pipeline(self, input_data: pd.DataFrame) -> pd.DataFrame:
        self.load_models(); df_agg = self.aggregate_data(self.expand_quantity(self.process_categories(self.preprocess_data(input_data)))); one_hot = self.one_hot_encode(df_agg); final_data = self.prepare_final_data(df_agg, one_hot); return self.predict(final_data)
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy(); required_cols = ['NO_EOR', 'CONTAINER_SIZE', 'CONTAINER_GRADE', 'DAMAGE', 'REPAIRACTION', 'COMPONENT', 'LOCATION', 'QTY']; missing_cols = set(required_cols) - set(df.columns)
        if missing_cols: raise ValueError(f"Kolom wajib tidak ditemukan: {missing_cols}")
        df['CONTAINER_TYPE'] = (df['CONTAINER_SIZE'].astype(str) + df['CONTAINER_GRADE'].astype(str)).astype("category"); df['LOCATION'] = (df['LOCATION'].astype(str).str.replace(r'(\d+\s*(?:st|nd|rd|th)?)\s*,\s*(\d+\s*(?:st|nd|rd|th)?)', r'\1-\2', flags=re.IGNORECASE, regex=True).str.upper().str.replace(r'\s*-\s*', '-', regex=True).str.replace(r'\.$', '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip())
        df["DAMAGE_COMPONENT"] = df["DAMAGE"].astype(str) + "_" + df["COMPONENT"].astype(str); df["DAMAGE_ACTION"] = df["DAMAGE"].astype(str) + "_" + df["REPAIRACTION"].astype(str); df["COMPONENT_ACTION"] = df["COMPONENT"].astype(str) + "_" + df["REPAIRACTION"].astype(str); df["LOCATION_DAMAGE"] = df["LOCATION"].astype(str) + "_" + df["DAMAGE"].astype(str)
        return df
    def process_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy(); columns = ['COMPONENT', 'LOCATION', 'DAMAGE', 'REPAIRACTION', 'DAMAGE_ACTION', 'DAMAGE_COMPONENT', 'COMPONENT_ACTION', 'LOCATION_DAMAGE']
        for depo in df["DEPO"].unique():
            if depo not in self.category_maps: continue
            mask, category_map = df["DEPO"] == depo, self.category_maps[depo]
            for col in columns: valid_categories = category_map.get(col, []); condition = ~df.loc[mask, col].isin(valid_categories); df.loc[mask & condition, col] = 'other'
        return df
    def expand_quantity(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy(); list_cols = ["COMPONENT", "LOCATION", "DAMAGE", "REPAIRACTION", "DAMAGE_ACTION", "DAMAGE_COMPONENT", "COMPONENT_ACTION", "LOCATION_DAMAGE"]; df["QTY"] = df["QTY"].astype("int8")
        for col in list_cols:
            if col in df.columns: col_array, qty_array = df[col].to_numpy(), df["QTY"].to_numpy(); df[col] = [[val] * qty if qty > 0 else [] for val, qty in zip(col_array, qty_array)]
        return df
    def aggregate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        first_cols, list_cols = ["NO_EOR", "CONTAINER_GRADE", "CONTAINER_SIZE", "CONTAINER_TYPE", "DEPO"], ["COMPONENT", "LOCATION", "DAMAGE", "REPAIRACTION", "DAMAGE_ACTION", "DAMAGE_COMPONENT", "COMPONENT_ACTION", "LOCATION_DAMAGE"]
        def flatten(series): return list(chain.from_iterable(v if isinstance(v, list) else [v] for v in series.dropna()))
        agg_dict = {col: "first" for col in first_cols if col != "NO_EOR"}; agg_dict.update({col: flatten for col in list_cols}); return data.groupby("NO_EOR", as_index=False, observed=True).agg(agg_dict)
    def one_hot_encode(self, data: pd.DataFrame) -> pd.DataFrame:
        list_cols, results = ["COMPONENT", "LOCATION", "DAMAGE", "REPAIRACTION", "DAMAGE_ACTION", "DAMAGE_COMPONENT", "COMPONENT_ACTION", "LOCATION_DAMAGE"], {}
        def safe_parse_list(val):
            if isinstance(val, str):
                try: return ast.literal_eval(val)
                except Exception: return []
            return val
        for column in list_cols:
            df_col = data[["NO_EOR", column]].copy(); df_col[column] = df_col[column].apply(safe_parse_list); df_exploded = df_col.explode(column).dropna(subset=[column]); dummies = pd.get_dummies(df_exploded[column], prefix=column); df_encoded = pd.concat([df_exploded[["NO_EOR"]], dummies], axis=1); results[column] = df_encoded.groupby("NO_EOR", as_index=False).sum()
        if not results: return pd.DataFrame({'NO_EOR': data['NO_EOR']})
        return reduce(lambda left, right: pd.merge(left, right, on="NO_EOR", how="outer"), list(results.values()))
    def prepare_final_data(self, data: pd.DataFrame, one_hot: pd.DataFrame) -> pd.DataFrame:
        df_agg = data.drop(columns=[col for col in ["COMPONENT", "LOCATION", "DAMAGE", "REPAIRACTION", "DAMAGE_ACTION", "DAMAGE_COMPONENT", "COMPONENT_ACTION", "LOCATION_DAMAGE"] if col in data.columns]); df_merged = pd.merge(df_agg, one_hot, on="NO_EOR", how="left")
        fill_values = {col: 0 for col in [c for c in one_hot.columns if c != 'NO_EOR'] if col in df_merged.columns}; return df_merged.fillna(value=fill_values)
    def predict(self, final_data: pd.DataFrame) -> pd.DataFrame:
        if "DEPO" not in final_data.columns: raise ValueError("DEPO column not found")
        all_results = []
        for depo in final_data["DEPO"].unique():
            if depo not in self.models: continue
            df_depo = final_data[final_data["DEPO"] == depo].copy(); model, expected_features = self.models[depo], self.models[depo].feature_names_
            base_df_for_pred = df_depo.drop(columns=["NO_EOR", "DEPO"]); missing_cols = set(expected_features) - set(base_df_for_pred.columns)
            for col in missing_cols: base_df_for_pred[col] = 0
            for col in self.cat_cols:
                if col in base_df_for_pred.columns: base_df_for_pred[col] = base_df_for_pred[col].astype(str)
            base_df_for_pred = base_df_for_pred[expected_features].copy()
            for vendor in self.depo_config[depo]["vendors"]:
                df_copy = base_df_for_pred.copy(); df_copy["IDKONTRAKTOR"] = vendor; pool = Pool(data=df_copy, cat_features=self.cat_cols); preds = model.predict(pool)
                valid_grades = self.validity_map.get(depo, {}).get(vendor, []); mask_invalid = ~df_depo['CONTAINER_GRADE'].isin(valid_grades)
                preds[mask_invalid] = np.nan; df_depo[f"PREDIKSI_{vendor}"] = preds
            all_results.append(df_depo)
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# ==============================================================================
# UI STREAMLIT DAN LOGIKA APLIKASI
# ==============================================================================
st.set_page_config(page_title="Container Repair Prediction", layout="wide")
st.title("📦 Alat Bantu Keputusan Perbaikan Kontainer")

# --- Kontrol Global di Sidebar ---
with st.sidebar:
    st.header("⚙️ Parameter Global")
    depo_option = st.selectbox("Pilih DEPO", ["JKT", "SBY"], key="global_depo")
    
# --- Struktur Tab Utama ---
tab_manual, tab_bulk = st.tabs(["Cek Harga Manual", "Alokasi Optimal (Upload CSV)"])

# ==============================================================================
# TAB 1: CEK HARGA MANUAL
# ==============================================================================
with tab_manual:
    st.header("Cek Estimasi Harga per Kerusakan (Manual)")
    st.info("Masukkan detail perbaikan untuk satu kontainer untuk melihat perbandingan harga antar vendor di DEPO yang dipilih.")

    st.subheader("📝 Masukkan Detail Kerusakan")
    num_entries = st.number_input("Jumlah Item Kerusakan", min_value=1, max_value=20, value=3, help="Tentukan berapa banyak baris kerusakan yang akan Anda masukkan.")

    @st.cache_data
    def load_unique_values():
        try:
            # Ganti dengan path yang benar jika file tidak di direktori yang sama
            return pd.read_csv("list_unique.csv")
        except FileNotFoundError:
            st.error("Error: 'list_unique.csv' tidak ditemukan. Menggunakan daftar kosong.")
            return pd.DataFrame({'DAMAGE': [], 'REPAIRACTION': [], 'LOCATION': [], 'COMPONENT': []})

    unique_list_df = load_unique_values()
    
    # Menambahkan placeholder di awal untuk memastikan user memilih
    DAMAGE_OPTIONS = ["- Pilih Tipe Kerusakan -"] + unique_list_df["DAMAGE"].dropna().unique().tolist()
    REPAIR_OPTIONS = ["- Pilih Tindakan -"] + unique_list_df["REPAIRACTION"].dropna().unique().tolist()
    LOCATION_OPTIONS = ["- Pilih Lokasi -"] + unique_list_df["LOCATION"].dropna().unique().tolist()
    COMPONENT_OPTIONS = ["- Pilih Komponen -"] + unique_list_df["COMPONENT"].dropna().unique().tolist()

    with st.form("manual_entry_form"):
        container_grade = st.selectbox("Container Grade", ['A', 'B', 'C'], help="Grade kontainer.", key="manual_grade")
        container_size = st.selectbox("Container Size", ['20', '40'], help="Ukuran kontainer (20ft atau 40ft).", key="manual_size")
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

        submitted = st.form_submit_button("🚀 Cek Estimasi Harga")

    if submitted:
        # Validasi input: pastikan tidak ada placeholder yang terpilih
        if any(opt.startswith("- Pilih") for opt in damage_data['damage']) or \
           any(opt.startswith("- Pilih") for opt in damage_data['repair_action']) or \
           any(opt.startswith("- Pilih") for opt in damage_data['location']) or \
           any(opt.startswith("- Pilih") for opt in damage_data['component']):
            st.warning("Mohon pastikan semua detail kerusakan telah dipilih dari dropdown.")
        else:
            with st.spinner("Menjalankan prediksi untuk input manual..."):
                manual_input_rows = []
                for i in range(num_entries):
                    manual_input_rows.append({
                        "NO_EOR": "MANUAL_CHECK", "CONTAINER_SIZE": container_size, "CONTAINER_GRADE": container_grade,
                        "DAMAGE": damage_data['damage'][i], "REPAIRACTION": damage_data['repair_action'][i],
                        "COMPONENT": damage_data['component'][i], "LOCATION": damage_data['location'][i],
                        "QTY": damage_data['qty'][i], "DEPO": depo_option
                    })
                manual_df = pd.DataFrame(manual_input_rows)
                
                try:
                    pipeline = ContainerRepairPipeline()
                    prediction_result = pipeline.run_pipeline(manual_df)
                    
                    if not prediction_result.empty:
                        result_row = prediction_result.iloc[0]
                        pred_cols = [col for col in prediction_result.columns if col.startswith("PREDIKSI_")]
                        price_series = result_row[pred_cols].dropna().sort_values()
                        
                        if not price_series.empty:
                            price_df = price_series.reset_index()
                            price_df.columns = ["Vendor", "Estimasi Harga"]
                            price_df["Vendor"] = price_df["Vendor"].str.replace("PREDIKSI_", "")
                            
                            st.subheader(f"Hasil Estimasi untuk DEPO {depo_option}")
                            st.dataframe(price_df.style.format({'Estimasi Harga': 'Rp {:,.0f}'}), use_container_width=True)
                        else:
                            st.error("Tidak ada vendor yang valid untuk kombinasi Grade dan DEPO yang dipilih.")
                    else:
                        st.error("Gagal mendapatkan hasil prediksi.")
                except Exception as e:
                    st.error(f"Terjadi error saat prediksi manual: {e}")

# ==============================================================================
# TAB 2: ALOKASI OPTIMAL (BULK)
# ==============================================================================
with tab_bulk:
    st.header("Alokasi Optimal untuk Banyak Kontainer")

    with st.expander("Opsi & Upload File", expanded=True):
        uploaded_file = st.file_uploader("Upload file CSV Anda", type=["csv"], key="bulk_upload")
        
        use_capacity_filter = st.checkbox("Gunakan filter kapasitas vendor", key="bulk_capacity_check")
        vendor_capacities_input = {}
        if use_capacity_filter:
            st.write("**Masukkan Kapasitas Vendor:**")
            temp_pipeline = ContainerRepairPipeline()
            vendors_in_depo = temp_pipeline.depo_config.get(depo_option, {}).get("vendors", [])
            for vendor in vendors_in_depo:
                capacity = st.number_input(f"Kapasitas {vendor}", min_value=0, value=100, key=f"capacity_bulk_{depo_option}_{vendor}")
                vendor_capacities_input[vendor] = capacity
        
        run_bulk_button = st.button("Jalankan Alokasi Optimal", type="primary", key="bulk_run")

    @st.cache_data
    def run_priority_queue_allocation(uploaded_file_content, depo_option, vendor_capacities_dict):
        data = pd.read_csv(StringIO(uploaded_file_content))
        data["DEPO"] = depo_option
        
        pipeline = ContainerRepairPipeline()
        raw_results = pipeline.run_pipeline(data).set_index('NO_EOR')
        pred_cols = [col for col in raw_results.columns if col.startswith("PREDIKSI_")]
        
        capacities = vendor_capacities_dict.copy()
        allocations = {}
        priority_queue = []

        # Fungsi helper untuk menghitung OC dari daftar harga
        def calculate_oc(prices_list):
            if len(prices_list) >= 2:
                return prices_list[1][1] - prices_list[0][1]
            return 0

        # Langkah 1: Inisialisasi Priority Queue
        for eor, row in raw_results.iterrows():
            prices = row[pred_cols].dropna().sort_values()
            sorted_vendor_options = [(p.replace('PREDIKSI_', ''), val) for p, val in prices.items()]
            
            if not sorted_vendor_options:
                continue
                
            opportunity_cost = calculate_oc(sorted_vendor_options)
            heapq.heappush(priority_queue, (-opportunity_cost, eor, sorted_vendor_options))

        # Langkah 2: Loop Alokasi Cerdas
        while priority_queue:
            neg_oc, eor, current_options = heapq.heappop(priority_queue)
            
            if eor in allocations:
                continue
                
            best_vendor, best_price = current_options[0]
            
            # Jika vendor termurah masih punya kapasitas, alokasikan
            if capacities.get(best_vendor, float('inf')) > 0:
                allocations[eor] = {
                    'REKOMENDASI': best_vendor, 
                    'HARGA_REKOMENDASI': best_price,
                    'SELISIH_SAAT_ALOKASI': -neg_oc
                }
                if best_vendor in capacities:
                    capacities[best_vendor] -= 1
            
            # Jika TIDAK, hitung ulang prioritas dan masukkan kembali ke antrian
            else:
                remaining_options = current_options[1:]
                if remaining_options:
                    new_oc = calculate_oc(remaining_options)
                    heapq.heappush(priority_queue, (-new_oc, eor, remaining_options))
                else:
                    allocations[eor] = {
                        'REKOMENDASI': 'TIDAK TERHANDLE', 
                        'HARGA_REKOMENDASI': np.nan,
                        'SELISIH_SAAT_ALOKASI': -neg_oc
                    }

        # Finalisasi hasil
        allocations_df = pd.DataFrame.from_dict(allocations, orient='index')
        final_df = raw_results.join(allocations_df, how='left')
        
        final_df['REKOMENDASI'] = final_df['REKOMENDASI'].fillna('TIDAK TERHANDLE')
        final_df['HARGA_REKOMENDASI'] = final_df['HARGA_REKOMENDASI'].fillna(np.nan)
        final_df['SELISIH_SAAT_ALOKASI'] = final_df['SELISIH_SAAT_ALOKASI'].fillna(0)

        return final_df.reset_index()

    # Main app logic
    if run_bulk_button and uploaded_file is not None:
        try:
            uploaded_file_content = uploaded_file.getvalue().decode('utf-8')
            capacities_to_use = vendor_capacities_input if use_capacity_filter else {}
            with st.spinner('Menjalankan alokasi cerdas dengan Priority Queue...'):
                final_results = run_priority_queue_allocation(uploaded_file_content, depo_option, capacities_to_use)
            
            st.success("✅ Alokasi optimal berhasil diselesaikan!")
            
            st.markdown("---")
            st.subheader("Ringkasan Hasil Alokasi")

            unhandled_count = (final_results['REKOMENDASI'] == 'TIDAK TERHANDLE').sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Kontainer", final_results['NO_EOR'].nunique())
            col2.metric("Total Pengeluaran", f"Rp {final_results['HARGA_REKOMENDASI'].sum():,.0f}")
            col3.metric("Kontainer Tidak Terhandle", f"{unhandled_count}")
            
            if unhandled_count > 0:
                st.warning(f"{unhandled_count} kontainer tidak dapat dialokasikan karena kapasitas penuh.")

            st.write("#### Distribusi Pekerjaan per Vendor")
            vendor_stats = final_results.groupby('REKOMENDASI').agg(Jumlah_Kontainer=('NO_EOR', 'nunique'), Total_Biaya=('HARGA_REKOMENDASI', 'sum')).reset_index().rename(columns={'REKOMENDASI': 'VENDOR'})
            st.dataframe(vendor_stats.style.format({'Total_Biaya': 'Rp {:,.0f}'}), use_container_width=True)

            st.markdown("---")
            st.subheader("Detail Hasil Alokasi dan Semua Prediksi Harga")
            
            pred_cols = [col for col in final_results.columns if col.startswith("PREDIKSI_")]
            display_cols = ['NO_EOR', 'CONTAINER_TYPE', 'REKOMENDASI', 'HARGA_REKOMENDASI', 'SELISIH_SAAT_ALOKASI'] + pred_cols
            
            display_df = final_results[[col for col in display_cols if col in final_results.columns]].copy()
            display_df.rename(columns={
                'REKOMENDASI': 'VENDOR_DIALOKASIKAN', 
                'HARGA_REKOMENDASI': 'HARGA_FINAL',
                'SELISIH_SAAT_ALOKASI': 'INCREMENT_GAP_SAAT_ALOKASI'
            }, inplace=True)
            display_df = display_df.sort_values(by="INCREMENT_GAP_SAAT_ALOKASI", ascending = False)
            
            def highlight_allocated_vendor(row):
                allocated_vendor = row['VENDOR_DIALOKASIKAN']
                if pd.isna(allocated_vendor) or allocated_vendor == 'TIDAK TERHANDLE': return [''] * len(row)
                highlight_col, styles = f"PREDIKSI_{allocated_vendor}", [''] * len(row)
                if highlight_col in row.index:
                    col_idx = row.index.get_loc(highlight_col)
                    styles[col_idx] = 'background-color: #D4EDDA; font-weight: bold; color: #155724;'
                return styles

            st.dataframe(
                display_df.style.format({'INCREMENT_GAP_SAAT_ALOKASI': 'Rp {:,.0f}'}, na_rep='-', precision=0)
                        .apply(highlight_allocated_vendor, axis=1), 
                height=600
            )
            
            csv_final = display_df[["NO_EOR", "CONTAINER_TYPE", "VENDOR_DIALOKASIKAN","HARGA_FINAL", "INCREMENT_GAP_SAAT_ALOKASI"]].to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Hasil Lengkap", data=csv_final, file_name="hasil_alokasi_lengkap.csv", mime="text/csv")
            
            if use_capacity_filter:
                st.info(f"Kapasitas awal yang digunakan dalam alokasi: {capacities_to_use}")

        except Exception as e:
            st.error(f"Terjadi error: {str(e)}"); st.exception(e)

    elif run_bulk_button:
        st.warning("Mohon unggah file CSV terlebih dahulu.")
    else:
        st.info("Silakan unggah file CSV dan klik 'Jalankan Prediksi & Alokasi' untuk memulai.")

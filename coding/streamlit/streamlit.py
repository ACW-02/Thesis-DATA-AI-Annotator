import streamlit as st
import pandas as pd
import json
import openai
import os
from math import ceil
import io

# ==========================================
# MEMUAT API KEY DARI STREAMLIT SECRETS
# ==========================================
# Use st.secrets instead of load_dotenv and os.getenv
try:
    API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    API_KEY = None

# ==========================================
# KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(page_title="AI Text Annotator", page_icon="📝", layout="wide")

# CSS Diperbarui: Menghapus .step-container HTML karena kita akan pakai container native Streamlit
st.markdown("""
    <style>
        .main-title { text-align: center; font-size: 3rem; font-weight: 800; color: #E53935; margin-top: 10px; margin-bottom: 0px; }
        .sub-title { text-align: center; font-size: 1.1rem; color: gray; margin-bottom: 40px; font-weight: 500; }
        
        .step-title { font-weight: 700; color: #E53935; margin-bottom: 10px; font-size: 1.2rem; border-bottom: 2px solid #E53935; padding-bottom: 5px; display: inline-block;}
        
        .stButton>button { background-color: #E53935; color: white; font-weight: bold; border-radius: 8px; height: 55px; font-size: 1.1rem; transition: 0.3s; margin-top: 20px;}
        .stButton>button:hover { background-color: #D32F2F; color: white; border-color: #D32F2F; }
        
        .stDownloadButton>button { background-color: #43A047; color: white; font-weight: bold; border-radius: 8px; height: 50px; width: 100%; transition: 0.3s; }
        .stDownloadButton>button:hover { background-color: #388E3C; color: white; border-color: #388E3C; }
    </style>
""", unsafe_allow_html=True)

# Judul & Subjudul Aplikasi
st.markdown("<div class='main-title'>AI Data Annotator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Klasifikasi teks pintar terotomasi dengan pilihan model GPT (Terbaru & Legacy) dan dukungan preset kustom.</div>", unsafe_allow_html=True)

# ==========================================
# PRESET BAWAAN: DATE MEMBER EXIT
# ==========================================
DATE_CATEGORIES = {
    "Admin": "Kesalahan administratif atau perubahan data (human error, false positive).",
    "lokasi": "Perpindahan lokasi geografis secara eksplisit (pindah kota/luar negeri).",
    "perbedaan_musim_kehidupan": "Perubahan fase kehidupan jangka panjang (menikah, studi).",
    "tidak_ada_respon": "Anggota tidak atau minim memberikan respons saat dihubungi.",
    "waktu_tidak_sesuai": "Benturan jadwal atau komitmen waktu rutin (jam kerja/kuliah).",
    "tertanam_di_gereja_lain": "Memilih untuk tetap tertanam atau beribadah di gereja lain.",
    "perbedaan_umur": "Ketidaksesuaian akibat perbedaan rentang usia di kelompok.",
    "alasan_DATE": "Dinamika internal DATE (ketidakcocokan, konflik, DATE bubar).",
    "others": "Alasan ambigu, terlalu singkat, wafat, atau tidak jelas."
}

DATE_SYS_PROMPT = "You are a data annotation system."
DATE_RULES = """Choose EXACTLY ONE category per text.
Label MUST match one of the category keys.
If relocation is explicitly mentioned → choose "lokasi".
If routine schedule conflict → choose "waktu_tidak_sesuai".
If long-term life phase change → choose "perbedaan_musim_kehidupan".
If related to DATE group condition/conflict → choose "alasan_DATE".
If unclear or insufficient information → choose "others".
Do not infer beyond the text.
Return a JSON object containing an array named "results" (same order as input).
Each object inside the "results" array must contain:
final_label (string)
confidence (0–100 integer)
keywords (Max 3 short keywords)
Output ONLY valid JSON. No explanations."""

DATE_USER_CONTENT = "Here is a list of {length} texts that need to be analyzed:\n{texts}"

# ==========================================
# FUNGSI KLASIFIKASI GPT 
# ==========================================
def classify_batch_with_gpt(batch_texts, api_key, model_name, sys_prompt, rules, categories_dict, user_content_temp):
    client = openai.OpenAI(api_key=api_key, timeout=120.0, max_retries=3)
    full_system_prompt = f"{sys_prompt}\n\nCategories: {json.dumps(categories_dict, ensure_ascii=False)}\n\nRules:\n{rules}"
    
    texts_formatted = ""
    for idx, text in enumerate(batch_texts):
        texts_formatted += f"[{idx+1}] {text}\n"

    user_msg = user_content_temp.replace("{length}", str(len(batch_texts))).replace("{texts}", texts_formatted)

    try:
        response = client.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=0,
            max_completion_tokens=10000,
            response_format={"type": "json_object"} 
        )
        raw_json = response.choices[ 0 ].message.content
        parsed_json = json.loads(raw_json)
        return parsed_json.get("results", [])
    except Exception as e:
        st.error(f"API Error ({model_name}): {e}")
        return []

# ==========================================
# ANTARMUKA PENGGUNA (UI) - PEMBAGIAN 2 KOLOM RAPI
# ==========================================
left_col, right_col = st.columns(2)

# ----------------- KOLOM KIRI -----------------
with left_col:
    # LANGKAH 1: Kredensial menggunakan Native Border Container
    with st.container(border=True):
        st.markdown("<div class='step-title'>🔑 1. Kredensial API</div>", unsafe_allow_html=True)
        if API_KEY:
            st.success("✅ OpenAI API Key terhubung")
        else:
            st.error("❌ OpenAI API Key tidak ditemukan")

    # LANGKAH 2: Pilih Model menggunakan Native Border Container
    with st.container(border=True):
        st.markdown("<div class='step-title'>🤖 2. Pemilihan Model</div>", unsafe_allow_html=True)
        model_options = {
        "GPT-5.4 Mini (Eksperimental)": "gpt-5.4-mini",
        "GPT-5.4 Nano (Paling Cepat, Eksperimental)": "gpt-5.4-nano",
        "GPT-4o Mini (Terbaru, Cepat & Efisien)": "gpt-4o-mini",
        "GPT-4o (Terbaru, Paling Cerdas)": "gpt-4o",
        "GPT-4 Turbo": "gpt-4-turbo",
        "GPT-3.5 Mini": "gpt-3.5-mini",
        "GPT-3.5 Turbo (Legacy)": "gpt-3.5-turbo"
    }
        selected_model_label = st.selectbox("Pilih arsitektur model OpenAI:", list(model_options.keys()))
        active_model = model_options[selected_model_label]

        # LANGKAH 3: Preset menggunakan Native Border Container
    with st.container(border=True):
        st.markdown("<div class='step-title'>⚙️ 3. Konfigurasi Preset</div>", unsafe_allow_html=True)
        preset_choice = st.selectbox("Pilih Preset Aturan Kategorisasi:", ["DATE Member exit", "+ Add preset"])
        custom_preset_data = None
        if preset_choice == "+ Add preset":
            json_file = st.file_uploader("Upload Preset Label (JSON)", type=["json"])
            if json_file:
                try:
                    custom_preset_data = json.load(json_file)
                    st.success("✅ Preset kustom berhasil dimuat!")
                except Exception as e:
                    st.error(f"❌ Gagal membaca JSON: {e}")
        else:
            st.info("✅ Menggunakan preset bawaan: Evaluasi alasan keluar DATE JPCC.")

# ----------------- KOLOM KANAN -----------------
with right_col:
    # LANGKAH 4: Upload Data menggunakan Native Border Container
    with st.container(border=True):
        st.markdown("<div class='step-title'>📄 4. Upload Data Mentah</div>", unsafe_allow_html=True)
        data_file = st.file_uploader("Pilih file data (CSV / XLSX)", type=["csv", "xlsx"])

        df = None
        text_col = None

        if data_file:
            if data_file.name.endswith('.csv'):
                df = pd.read_csv(data_file)
            else:
                df = pd.read_excel(data_file)
            
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                text_col = st.selectbox("Kolom Target Analisis", df.columns.tolist(), index=df.columns.tolist().index("delete_reason") if "delete_reason" in df.columns else 0, help="Pilih kolom pada file Excel/CSV Anda yang memuat teks narasi/alasan jemaat yang ingin dianalisis oleh AI.")
            with sub_col2:
                user_batch_size = st.number_input(label="Ukuran Batch (Jumlah Baris)", value=10, help="Menentukan berapa banyak baris data yang dikirim ke AI dalam satu waktu.\n\n"
         "Panduan Pengaturan:\n"
         "⬇️ Angka Kecil (5-10): Lebih stabil dan anti-error (timeout). Gunakan jika teks panjang-panjang.\n"
         "⬆️ Angka Besar (50-100): Lebih cepat selesai. Gunakan hanya jika teks pendek-pendek dan koneksi lancar.")

# ==========================================
# TOMBOL EKSEKUSI (LEBAR PENUH)
# ==========================================
if st.button("🚀 Mulai Anotasi Sekarang", use_container_width=True):
    if not API_KEY:
        st.warning("⚠️ Mohon pastikan API Key telah dikonfigurasi di file .env Anda.")
    elif preset_choice == "+ Add preset" and custom_preset_data is None:
        st.warning("⚠️ Mohon upload file JSON preset kustom Anda.")
    elif df is None or text_col is None:
        st.warning("⚠️ Mohon upload file data mentah (CSV/XLSX) yang ingin dianotasi.")
    else:
        with st.status(f"Menganotasi data Anda menggunakan **{active_model}**...", expanded=True) as status:
            try:
                # 1. Setup Preset
                if preset_choice == "DATE Member exit":
                    cat_dict = DATE_CATEGORIES
                    sys_pr = DATE_SYS_PROMPT
                    rules_pr = DATE_RULES
                    user_cont = DATE_USER_CONTENT
                else:
                    cat_dict = custom_preset_data.get("category_definition", {})
                    sys_pr = custom_preset_data.get("system_prompt", "You are an AI annotator.")
                    rules_pr = custom_preset_data.get("rules", "Return a JSON object with a 'results' array.")
                    user_cont = custom_preset_data.get("user_content", "Texts:\n{texts}")

                # 2. Batching Setup
                batch_size = user_batch_size
                num_batches = ceil(len(df) / batch_size)
                annotated_results = []
                progress_bar = st.progress(0)

                # 3. Proses LLM
                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(df))
                    batch = df.iloc[start:end]
                    
                    st.write(f"Menganotasi batch {batch_idx + 1} dari {num_batches}...")
                    
                    outputs = classify_batch_with_gpt(
                        batch[text_col].astype(str).tolist(), 
                        API_KEY, 
                        active_model,
                        sys_pr, 
                        rules_pr, 
                        cat_dict, 
                        user_cont
                    )
                    
                    for i, (_, row) in enumerate(batch.iterrows()):
                        out = outputs[i] if i < len(outputs) else {}
                        
                        row_dict = row.to_dict()
                        row_dict["AI_Label"] = out.get("final_label", "others")
                        row_dict["AI_Confidence"] = f"{out.get('confidence', 0)}%"
                        row_dict["AI_Keywords"] = ", ".join(out.get("keywords", [])) if out.get("keywords") else ""
                        annotated_results.append(row_dict)
                    
                    progress_bar.progress((batch_idx + 1) / num_batches)
                
                status.update(label="Anotasi Selesai!", state="complete", expanded=False)
                
                # 4. Preview & Download
                st.success("🎉 Data berhasil dianotasi sepenuhnya!")
                df_results = pd.DataFrame(annotated_results)
                st.dataframe(df_results, use_container_width=True)

                output_buffer = io.BytesIO()
                with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                    df_results.to_excel(writer, index=False, sheet_name='Annotated_Data')
                
                st.download_button(
                    label="⬇️ Unduh Hasil Anotasi (XLSX)",
                    data=output_buffer.getvalue(),
                    file_name="Annotated_Results_Final.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            except Exception as e:
                status.update(label="Terjadi Kesalahan", state="error")
                st.error(f"Error: {e}")

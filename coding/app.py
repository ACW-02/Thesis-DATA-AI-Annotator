import app as st
import pandas as pd
import json
import openai
from math import ceil
import io

# api_key="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"
# ==========================================
# KONFIGURASI HALAMAN & CSS (gaya iLovePDF)
# ==========================================
st.set_page_config(page_title="DATE Exit Reason Annotator", page_icon="📝", layout="centered")

st.markdown("""
    <style>
        /* Mengatur judul bergaya iLovePDF (Besar, Bold, Terpusat) */
        .main-title { text-align: center; font-size: 3rem; font-weight: 800; color: #E53935; margin-top: 20px; margin-bottom: 0px; }
        .sub-title { text-align: center; font-size: 1.2rem; color: #555555; margin-bottom: 40px; }
        
        /* Mempercantik tombol utama */
        .stButton>button { background-color: #E53935; color: white; font-weight: bold; border-radius: 8px; height: 50px; }
        .stButton>button:hover { background-color: #D32F2F; color: white; }
        
        /* Mempercantik tombol download */
        .stDownloadButton>button { background-color: #43A047; color: white; font-weight: bold; border-radius: 8px; }
        .stDownloadButton>button:hover { background-color: #388E3C; color: white; }
    </style>
""", unsafe_allow_html=True)

# Judul Aplikasi
st.markdown("<div class='main-title'>AI Data Annotator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Klasifikasi Otomatis Alasan Keluar Jemaat DATE (Powered by GPT-5.4-mini)</div>", unsafe_allow_html=True)

# ==========================================
# FUNGSI KLASIFIKASI (OPENAI GPT-5.4-MINI)
# ==========================================
def classify_batch_with_gpt(batch_texts, categories_dict, api_key):
    client = openai.OpenAI(api_key=api_key)
    
    # Sedikit penyesuaian: Meminta JSON object bernama "results" agar kompatibel sempurna dengan response_format OpenAI
    system_prompt = f"""You are a data annotation system. Categories: {json.dumps(categories_dict, ensure_ascii=False)}
    Rules:
    Choose EXACTLY ONE category per text.
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

    user_content = f"Here is a list of {len(batch_texts)} texts that need to be analyzed:\n"
    for idx, text in enumerate(batch_texts):
        user_content += f"[{idx+1}] {text}\n"

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0,
            max_completion_tokens=10000, # Memastikan limit token panjang
            response_format={"type": "json_object"} # Memaksa OpenAI mengembalikan JSON yang valid
        )

        raw_json = response.choices[ 0 ].message.content
        parsed_json = json.loads(raw_json)
        
        # Mengekstrak array dari dalam objek JSON
        return parsed_json.get("results", [])

    except Exception as e:
        st.error(f"API Error: {e}")
        return []

# ==========================================
# ANTARMUKA PENGGUNA (UI)
# ==========================================

# 1. Input API Key
api_key_input = st.text_input("🔑 Masukkan OpenAI API Key Anda:", type="password", help="API Key tidak akan disimpan.")

# 2. Area Drag & Drop File
col1, col2 = st.columns(2)
with col1:
    st.info("📄 **Langkah 1:** Upload Data Mentah")
    data_file = st.file_uploader("Drag & drop file CSV atau XLSX", type=["csv", "xlsx"])

with col2:
    st.info("🏷️ **Langkah 2:** Upload Definisi Kategori")
    json_file = st.file_uploader("Drag & drop file Label (JSON)", type=["json"])

# 3. Tombol Proses
if st.button("🚀 Mulai Anotasi Sekarang", use_container_width=True):
    if not api_key_input:
        st.warning("⚠️ Mohon masukkan API Key OpenAI Anda terlebih dahulu.")
    elif not data_file:
        st.warning("⚠️ Mohon upload file data yang ingin dianotasi.")
    elif not json_file:
        st.warning("⚠️ Mohon upload file JSON definisi kategori Anda.")
    else:
        with st.status("Sedang memproses data Anda dengan GPT-5.4-mini...", expanded=True) as status:
            try:
                # Membaca file JSON kategori
                st.write("Membaca file kategori...")
                categories_dict = json.load(json_file)
                
                # Membaca file Data (CSV atau Excel)
                st.write("Membaca data jemaat...")
                if data_file.name.endswith('.csv'):
                    df = pd.read_csv(data_file)
                else:
                    df = pd.read_excel(data_file)

                # Pastikan kolom delete_reason ada
                if "delete_reason" not in df.columns:
                    st.error("Error: File data Anda harus memiliki kolom bernama 'delete_reason'.")
                    st.stop()

                batch_size = 10
                num_batches = ceil(len(df) / batch_size)
                results = []
                
                # Progress bar
                progress_bar = st.progress(0)

                # Proses Batching
                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, len(df))
                    batch = df.iloc[start:end]
                    
                    st.write(f"Menganotasi batch {batch_idx + 1} dari {num_batches}...")
                    
                    outputs = classify_batch_with_gpt(
                        batch["delete_reason"].tolist(), 
                        categories_dict, 
                        api_key_input
                    )
                    
                    # Gabungkan hasil
                    for i, (_, row) in enumerate(batch.iterrows()):
                        out = outputs[i] if i < len(outputs) else {}
                        results.append({
                            "delete_reason": row["delete_reason"],
                            "Final label": out.get("final_label", "others"),
                            "Confidence": f"{out.get('confidence', 0)}%",
                            "Keywords": ", ".join(out.get("keywords", [])) if out.get("keywords") else "",
                        })
                    
                    # Update progress bar
                    progress_bar.progress((batch_idx + 1) / num_batches)
                
                # Selesai
                status.update(label="Anotasi Selesai!", state="complete", expanded=False)
                
                # Tampilkan Preview Data
                st.success("🎉 Data berhasil dianotasi dengan OpenAI!")
                st.subheader("👀 Preview Hasil Anotasi")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)

                # Tombol Download ke Excel (XLSX)
                output_buffer = io.BytesIO()
                with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                    df_results.to_excel(writer, index=False, sheet_name='Annotated_Data')
                
                st.download_button(
                    label="⬇️ Unduh File Excel (XLSX)",
                    data=output_buffer.getvalue(),
                    file_name="Annotated_DATE_Reasons_GPT.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            except Exception as e:
                status.update(label="Terjadi Kesalahan", state="error")
                st.error(f"Error: {e}")
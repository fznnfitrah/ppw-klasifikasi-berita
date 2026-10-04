import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import joblib
from gensim.models import Word2Vec
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================
# 1. KONFIGURASI CACHE STREAMLIT (AGAR CEPAT)
# ==========================================
@st.cache_resource
def load_nlp_tools():
    """Memuat Stemmer dan Stopwords Sastrawi hanya sekali di awal."""
    factory_stemmer = StemmerFactory()
    stemmer = factory_stemmer.create_stemmer()
    
    factory_stopword = StopWordRemoverFactory()
    daftar_stopword = factory_stopword.get_stop_words()
    
    return stemmer, daftar_stopword

@st.cache_resource
def load_models():
    """Memuat model Skip-gram dan Naive Bayes yang sudah dilatih."""
    # Pastikan file w2v_skenario1_final.model dan naive_bayes_skenario1.pkl satu folder dengan app.py
    # Ubah nama file di bawah ini jika Anda ingin menggunakan skenario lain
    w2v_model = Word2Vec.load("models/w2v_skenario1_final.model")
    nb_model = joblib.load("models/naive_bayes_skenario1.pkl")
    return w2v_model, nb_model

# ==========================================
# 2. FUNGSI SCRAPING DETIK.COM
# ==========================================
def scrape_detik_news(url):
    """Mengekstrak teks berita dari URL Detik.com dengan aturan general."""
    try:
        # Menambahkan headers agar tidak diblokir oleh sistem anti-bot
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return None, f"Gagal mengakses URL. Status code: {response.status_code}"
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # --- ATURAN PENCARIAN BERINGKAT (FALLBACK SYSTEM) ---
        
        # Rencana A: Mencari div yang memiliki class 'detail__body-text' (Paling spesifik & akurat)
        # Tanda titik (.) mewakili pencarian class dalam CSS Selector.
        # Ini mengabaikan class lain seperti 'font-google-sans-flex' dll.
        container = soup.select_one('div.detail__body-text')
        
        # Rencana B: Jika gagal, cari class 'itp_bodycontent'
        if not container:
            container = soup.select_one('div.itp_bodycontent')
            
        # Rencana C: Jika masih gagal, cari tag <article> secara umum
        if not container:
            container = soup.select_one('article')
            
        # Jika semua rencana gagal
        if not container:
            return None, "Struktur halaman sama sekali tidak dikenali atau ini bukan artikel berita standar."
            
        # Mengambil semua teks dari tag <p> dan menggabungkannya
        paragraphs = container.find_all('p')
        teks_berita = " ".join([p.get_text(strip=True) for p in paragraphs])
        
        # Validasi tambahan: memastikan teks tidak kosong (misal halamannya isinya cuma video/foto)
        if not teks_berita.strip():
            return None, "Halaman berhasil dibuka, tetapi tidak ada teks paragraf yang ditemukan."
            
        return teks_berita, "Sukses"
        
    except Exception as e:
        return None, f"Terjadi kesalahan saat scraping: {str(e)}"

# ==========================================
# 3. PIPELINE PREPROCESSING & EMBEDDING
# ==========================================
def preprocess_modular(teks, stemmer, daftar_stopword, hapus_angka=True):
    # 1. Case Folding & Hapus Tanda Baca
    teks = str(teks).lower()
    teks = re.sub(r'[^\w\s]', ' ', teks)
    
    # 2. Opsional: Hapus Angka (Default True untuk Skenario 1)
    if hapus_angka:
        teks = re.sub(r'\d+', '', teks)
        
    # 3. Tokenisasi
    tokens = teks.split()
    
    # 4. Hapus Stopword
    tokens = [w for w in tokens if w not in daftar_stopword]
        
    # 5. Stemming
    teks_gabung = " ".join(tokens)
    teks_stem = stemmer.stem(teks_gabung)
    tokens_final = teks_stem.split()
        
    return tokens_final

def get_vektor_rata_rata(tokens, model):
    vektor_kata = [model.wv[kata] for kata in tokens if kata in model.wv]
    if not vektor_kata:
        return np.zeros(model.vector_size)
    return np.mean(vektor_kata, axis=0)

# ==========================================
# 4. ANTARMUKA STREAMLIT (UI)
# ==========================================
st.set_page_config(page_title="Klasifikasi Berita Detik", page_icon="📰", layout="centered")

st.title("📰 Klasifikasi Berita Detik.com")
st.write("Aplikasi ini menggunakan **Word2Vec (Skip-gram)** dan **Naive Bayes** untuk mengklasifikasikan berita ke dalam kategori **Sport** atau **Finance**.")

# Inisialisasi resource di background
with st.spinner('Memuat komponen NLP dan Model Machine Learning...'):
    stemmer, daftar_stopword = load_nlp_tools()
    try:
        w2v_model, nb_model = load_models()
        model_siap = True
    except FileNotFoundError:
        st.error("⚠️ File model tidak ditemukan! Pastikan 'w2v_skenario1_final.model' dan 'naive_bayes_skenario1.pkl' berada di folder yang sama dengan app.py.")
        model_siap = False

if model_siap:
    url_input = st.text_input("Masukkan URL Berita Detik.com (Sport/Finance):", placeholder="https://finance.detik.com/...")
    
    if st.button("Analisis Berita", type="primary"):
        if not url_input.strip():
            st.warning("Silakan masukkan URL terlebih dahulu.")
        else:
            # Tahap 1: Scraping
            with st.status("Memproses URL...", expanded=True) as status:
                st.write("Mengunduh halaman web...")
                teks_asli, pesan_status = scrape_detik_news(url_input)
                
                if teks_asli:
                    st.write("Membersihkan dan melakukan tokenisasi teks (Preprocessing)...")
                    # Sesuaikan parameter hapus_angka dengan Skenario model yang Anda load
                    tokens = preprocess_modular(teks_asli, stemmer, daftar_stopword, hapus_angka=True)
                    
                    st.write("Mengekstrak fitur menggunakan Word2Vec Skip-gram...")
                    vektor_dokumen = get_vektor_rata_rata(tokens, w2v_model)
                    
                    st.write("Mengklasifikasikan dengan Naive Bayes...")
                    vektor_2d = np.array(vektor_dokumen).reshape(1, -1)
                    prediksi = nb_model.predict(vektor_2d)[0]
                    
                    status.update(label="Proses Selesai!", state="complete", expanded=False)
                    
                    # Menampilkan Hasil Prediksi
                    st.subheader("Hasil Klasifikasi:")
                    if prediksi.lower() == 'sport':
                        st.success(f"⚽ Berita ini diklasifikasikan sebagai: **SPORT**")
                    else:
                        st.info(f"📈 Berita ini diklasifikasikan sebagai: **FINANCE**")
                        
                    # Menampilkan detail ekstraksi
                    with st.expander("Lihat Detail Teks yang Diproses"):
                        st.markdown("**Teks Asli (Hasil Scraping):**")
                        st.write(teks_asli[:1000] + "..." if len(teks_asli) > 1000 else teks_asli)
                        
                        st.markdown("**Hasil Preprocessing (Tokens):**")
                        st.write(tokens[:50]) # Menampilkan 50 token pertama
                else:
                    status.update(label="Gagal memproses", state="error", expanded=False)
                    st.error(pesan_status)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------
# 📌 Judul Aplikasi
st.title("🌤️ Analisis Polusi Udara - Kota Wanliu 🌧️")

# ------------------------------
# Membaca File
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "PRSA_Data_Wanliu_20130301-20170228.csv")
    Wanliu_df = pd.read_csv(file_path)
    return Wanliu_df

Wanliu_df = load_data()

# ------------------------------
# Sidebar untuk Filter Data
st.sidebar.header("🔍 Filter Data")

# **Filter berdasarkan bulan tertentu**
bulan = Wanliu_df['month'].unique()
bulan_dipilih = st.sidebar.multiselect("Pilih Bulan", bulan, default=bulan)

# Terapkan filter ke DataFrame
filtered_Wanliu = Wanliu_df[Wanliu_df['month'].isin(bulan_dipilih)]

# Menampilkan ikon dan judul
st.markdown("### 📌 Data Awal")

# Membuat expander untuk menampilkan data
with st.expander(" **Lihat Data**", expanded=False):
    # Misalnya ingin menampilkan dataframe
    import pandas as pd
    file_path = os.path.join(os.path.dirname(__file__), "PRSA_Data_Wanliu_20130301-20170228.csv")
    Wanliu_df = pd.read_csv(file_path)
    st.dataframe(Wanliu_df)


# ------------------------------
# Hapus nilai Inf dan NaN
filtered_df = filtered_Wanliu.replace([float('inf'), float('-inf')], None).dropna()

# Cek ulang Missing Values setelah cleaning
cleaned_missing_values = filtered_Wanliu.isnull().sum()

# ------------------------------
# Deteksi Outlier dengan IQR
st.subheader("📌 Periksa dan Hapus Outlier")

kolom_numerik = ['PM2.5', 'PM10', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

Q1 = filtered_Wanliu[kolom_numerik].quantile(0.25)
Q3 = filtered_Wanliu[kolom_numerik].quantile(0.75)
IQR = Q3 - Q1

batas_bawah = Q1 - 1.5 * IQR
batas_atas = Q3 + 1.5 * IQR

clean_df = filtered_Wanliu[~((filtered_Wanliu[kolom_numerik] < batas_bawah) | (filtered_Wanliu[kolom_numerik] > batas_atas)).any(axis=1)]

# Cek jumlah data sebelum dan sesudah
st.write(f"Jumlah data sebelum pembersihan: {filtered_Wanliu.shape[0]}")
st.write(f"Jumlah data setelah pembersihan: {clean_df.shape[0]}")

# ------------------------------
# Exploratory Data Analysis (EDA)
st.subheader("📌 Exploratory Data Analysis (EDA)")

# Hitung rata-rata PM2.5 dan PM10 per bulan setelah filter
monthly_mean = clean_df.groupby('month')[kolom_numerik].mean().reset_index()

clean_df['RAIN'].fillna(clean_df['RAIN'].median(), inplace=True)

# Korelasi antar variabel yang dipilih
korelasi = clean_df[kolom_numerik].corr()

# ------------------------------
# Visualisasi Tren Musiman PM2.5 & PM10
st.subheader("📊 Tren Musiman Berdasarkan Filter")

fig, ax = plt.subplots(figsize=(12, 6))
for col in ['PM2.5', 'PM10']:
    sns.lineplot(x=monthly_mean['month'], y=monthly_mean[col], label=col, marker='o', ax=ax)

ax.set_xlabel('Bulan')
ax.set_ylabel('Konsentrasi Polutan')
ax.set_title('Tren Musiman Berdasarkan Filter')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Membuat expander untuk menampilkan data
with st.expander(" **Kesimpulan**", expanded=False):
    st.write("Terjadinya perubahan musim yang tidak menentu "
    "sepanjang tahun yang disebabkan oleh banyak faktor. PM merupakan "
    "Particulate Matter yang artinya semakin kecil nilainya (partikel "
    "udaranya) semakin berbahaya bagi kesehatan. Tren musiman PM10 "
    "cenderung lebih tinggi dibandingkan dengan PM2.5. "
    "Dengan asumsi musim panas terjadi pada bulan Juni-Agustus, "
    "maka pada musim panas tingkat polusi udara cenderung lebih rendah "
    "dibandingkan pada musim panas, begitu juga sebaliknya ketika musim hujan "
    "tingkat polutannya bertambah.")
    
# ------------------------------
#Cek variabel RAIN, jika kosong isi dengan median
clean_df['RAIN'].fillna(clean_df['RAIN'].median(), inplace=True)

clean_df['RAIN'] = clean_df['RAIN'].interpolate()

if clean_df['RAIN'].nunique() <= 1:  
    clean_df.drop(columns=['RAIN'], inplace=True)

# Hanya ambil kolom numerik yang benar-benar memiliki variasi data
kolom_valid = [col for col in kolom_numerik if col in clean_df.columns and clean_df[col].nunique() > 1]

# Hitung korelasi hanya dengan kolom yang valid
korelasi = clean_df[kolom_valid].corr()

# Heatmap Korelasi
st.subheader("📊 Korelasi antara Variabel yang Dipilih")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(korelasi, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Korelasi antara Variabel yang Dipilih')
st.pyplot(fig)

# ------------------------------
# Kesimpulan
st.subheader("📌 Kesimpulan")
with st.expander(" **Kesimpulan**", expanded=False):
    st.write("""Pengguna dapat menyesuaikan bulan untuk melihat pola polusi berdasarkan musim.
    Berdasarkan heatmap yang dihasilkan dari perhitungan, variabel TEMP (suhu udara) 
    memberikan nilai korelasi negatif yang artinya ketika suhu meningkat, 
    kadar PM2.5 cenderung menurun. Variabel PRES (tekanan udara) dan DEWP (titik embun) 
    tidak memberikan pengaruh besar karena nilainya yang sangat kecil. 
    Variabel RAIN(hujan), memiliki nilai korelasi negatif , sehingga ketika terjadi 
    hujan polusi udara akan menurun. Variabel WSPM(kecepatan angin) memiliki pengaruh 
    terhadap penyebaran polutan PM2.5 ini, semakin besar nilainya atau tinggi kecepatan 
    anginnya konsentrasi PM2.5 semakin rendah polutan akan tersebar luas.Heatmap korelasi 
    memperlihatkan hubungan antara PM2.5, PM10, dan faktor lainnya secara lebih interaktif.
""")
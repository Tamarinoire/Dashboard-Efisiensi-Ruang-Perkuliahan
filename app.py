import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(page_title="Dashboard Efisiensi Ruang Perkuliahan", page_icon="🏫", layout="wide")
# Styling: light theme (eye-friendly), card container, professional font
# st.markdown(

#     """
#     <style>
#       @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
#       .appview-container .main .block-container { max-width: 1200px; padding: 34px 48px; margin: 0 auto; font-family: 'Inter', Arial, sans-serif; }
#       .stApp { background: #ffffff; color: #0f172a; }
#       .card { background: #f8fafc; border-radius: 12px; padding: 18px; box-shadow: 0 6px 18px rgba(0,0,0,0.1); color: #0f172a; }
#       .dashboard-title { text-align:center; font-size:34px; font-weight:700; color:#0f172a; margin-bottom:6px; }
#       .dashboard-sub { text-align:center; color:#64748b; font-size:14px; margin-top:0; margin-bottom:18px; }
#       .stMetricValue, .stMetricValue > div { font-size:26px !important; font-weight:700; color:#0f172a !important; }
#       .stMetricLabel { font-size:13px !important; color:#64748b !important; }
#       /* Improve plotly container look */
#       .plotly-graph-div { background: transparent !important; }
#       /* Inputs */
#       .stSelectbox, .stTextInput, .stButton { font-family: 'Inter', Arial, sans-serif; color: #0f172a; }
#       .stSelectbox div[data-baseweb="select"] { background-color: #f8fafc !important; color: #0f172a !important; }
#       .stTextInput input { background-color: #f8fafc !important; color: #0f172a !important; }
#       .stButton button { background-color: #3b82f6 !important; color: #ffffff !important; }
#       /* Ensure all text is visible */
#       .stSubheader h3, .stMarkdown p, .stText, .stCaption, .stWarning, .stInfo, .stError { color: #0f172a !important; }
#       .stTable th, .stTable td { color: #0f172a !important; background-color: #f8fafc !important; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# Header (no emoji in title)
st.markdown('<div class="dashboard-title">Dashboard Efisiensi Ruang Perkuliahan</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-sub">Visualisasi hasil klasterisasi penggunaan ruang perkuliahan</div>', unsafe_allow_html=True)
st.markdown("---")

# Load data with fallback sample
@st.cache_data
def load_data(path="clustering_results.xlsx"):
    try:
        df = pd.read_excel(path)
    except Exception:
        df = pd.DataFrame({
            "Ruang": [f"R{n}" for n in range(1, 87)],
            "Cluster": [0]*30 + [2]*17 + [1]*39,
            "Gedung": ["GK 1"]*39 + ["Gedung E"]*17 + ["Gedung F"]*30,
            "Luas Ruangan": [50 + (n % 5)*10 for n in range(86)],
            "Jumlah AC": [1 + (n % 3) for n in range(86)],
            "Waktu penggunaan": [8 + (n % 4) for n in range(86)]
        })
    return df

df = load_data()

# Normalize and rename
df.columns = df.columns.str.strip()
rename_map = {
    "Ruang": "Nama_Ruangan",
    "Cluster": "cluster",
    "Waktu penggunaan": "Waktu_Penggunaan",
    "Waktu Penggunaan": "Waktu_Penggunaan",
    "Luas Ruangan": "Luas_Ruangan",
    "Jumlah AC": "Jumlah_AC"
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# Ensure essential columns
if "Nama_Ruangan" not in df.columns:
    df["Nama_Ruangan"] = df.index.astype(str)
if "cluster" not in df.columns:
    df["cluster"] = 0
if "Gedung" not in df.columns:
    df["Gedung"] = "Gedung F"
if "Luas_Ruangan" not in df.columns:
    df["Luas_Ruangan"] = 0
if "Jumlah_AC" not in df.columns:
    df["Jumlah_AC"] = 0

# Map clusters to labels
label_map = {0: "Kurang Efisien", 1: "Efisien", 2: "Sedang"}
try:
    df["cluster_num"] = pd.to_numeric(df["cluster"], errors="coerce")
    df["Kategori Efisiensi"] = df["cluster_num"].map(label_map)
    df["Kategori Efisiensi"] = df["Kategori Efisiensi"].fillna(df["cluster"].astype(str))
except Exception:
    df["Kategori Efisiensi"] = df["cluster"].astype(str)

# Distribusi Jumlah Kelas (Ruangan) Menurut Gedung: bar chart di paling atas
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Distribusi Jumlah Kelas (Ruangan) Menurut Gedung")
gedung_counts = df["Gedung"].value_counts().reset_index()
gedung_counts.columns = ["Gedung", "Jumlah Ruangan"]

color_map_gedung = {"GK 1": "#FF6B8A", "Gedung E": "#FFB347", "Gedung F": "#4ADE80"}

fig_bar = px.bar(
    gedung_counts,
    x="Gedung",
    y="Jumlah Ruangan",
    color="Gedung",
    color_discrete_map=color_map_gedung,
    template="plotly_white",
    height=400,
    text="Jumlah Ruangan"
)
fig_bar.update_traces(textposition="outside", textfont=dict(size=12, color="#0f172a"))
fig_bar.update_layout(
    showlegend=False,
    margin=dict(l=40, r=40, t=30, b=40),
    yaxis=dict(title="Jumlah Ruangan", tickfont=dict(size=12)),
    xaxis=dict(title="", tickfont=dict(size=13)),
    font=dict(family="Inter", size=13, color="#0f172a"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)
fig_bar.update_xaxes(tickangle=0)
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")  # spacing

# Select gedung
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Pilih Gedung")
gedung_options = ["Gedung F", "Gedung E", "GK 1"]
gedung_user = st.selectbox("Pilih Gedung", options=gedung_options, key="gedung_select")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")  # spacing

# Filter data berdasarkan gedung yang dipilih
df_gedung = df[df["Gedung"] == gedung_user]

# Top card with metrics untuk gedung yang dipilih
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"Ringkasan untuk {gedung_user}")
c1, c2, c3, c4 = st.columns([1.2,1,1,1])

total_rooms = df_gedung["Nama_Ruangan"].nunique()
count_kurang = (df_gedung["Kategori Efisiensi"] == "Kurang Efisien").sum()
count_sedang = (df_gedung["Kategori Efisiensi"] == "Sedang").sum()
count_efisien = (df_gedung["Kategori Efisiensi"] == "Efisien").sum()

c1.metric("Total Ruangan", f"{total_rooms}")
c2.metric("Kurang Efisien", f"{count_kurang}")
c3.metric("Sedang", f"{count_sedang}")
c4.metric("Efisien", f"{count_efisien}")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")  # spacing

# PIE / DONUT chart for distribution di gedung yang dipilih
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"Distribusi Kategori Efisiensi di {gedung_user} (Pie Chart)")
order = ["Kurang Efisien", "Sedang", "Efisien"]
counts = df_gedung["Kategori Efisiensi"].value_counts().reindex(order).fillna(0).astype(int).reset_index()
counts.columns = ["Kategori", "Count"]

color_map = {"Kurang Efisien": "#FF6B8A", "Sedang": "#FFB347", "Efisien": "#4ADE80"}

fig = px.pie(
    counts,
    names="Kategori",
    values="Count",
    color="Kategori",
    color_discrete_map=color_map,
    hole=0.38,
    template="plotly_white",
    height=420
)
fig.update_traces(textinfo="percent+label", textposition="inside", textfont=dict(size=14, family="Inter", color="#0f172a"))
fig.update_layout(
    margin=dict(l=40, r=40, t=30, b=40),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=13, color="#0f172a")
)
st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")  # spacing

# Search card untuk gedung yang dipilih
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"Cari Detail Ruangan di {gedung_user}")
ruangan_user = st.text_input("Nama Ruangan (partial search)", placeholder="Mis: F213 atau GK101")

if st.button("Cari Ruangan"):
    if ruangan_user.strip() == "":
        st.warning("Silakan masukkan nama ruangan!")
    else:
        mask = df_gedung["Nama_Ruangan"].astype(str).str.contains(ruangan_user, case=False, na=False)
        df_ruangan = df_gedung[mask]
        if df_ruangan.empty:
            st.info("Ruangan tidak ditemukan di gedung tersebut.")
        else:
            st.table(df_ruangan.loc[:, ["Nama_Ruangan", "Luas_Ruangan", "Jumlah_AC", "Kategori Efisiensi"]])

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Tampilan profesional — direkomendasikan layar minimal 1280×720.")



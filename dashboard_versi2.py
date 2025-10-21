# app.py
# ---------------------------------------------------------
# Colorful Survey Dashboard
# Fokus: "Pengaruh Teknologi dalam Proses Belajar Mahasiswa"
# ---------------------------------------------------------

import io
import os
import glob
import re
import textwrap
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# Page config & CSS
# =========================
st.set_page_config(page_title="Colorful Survey Dashboard", page_icon="📝", layout="wide")

CARD_CSS = """
<style>
.kpi-card{
  border-radius: 14px; padding: 18px 20px; color: #fff;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
  box-shadow: 0 6px 18px rgba(0,0,0,.08);
  border: 1px solid rgba(255,255,255,.15);
}
.kpi-title{font-size: 0.95rem; opacity: .9; margin-bottom: 4px;}
.kpi-value{font-size: 2rem; font-weight: 800; letter-spacing: .2px;}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.8rem;background:#0ea5e9;color:#fff}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

# ==== Sidebar radio: kotak full-width & tanpa bulatan ====
SIDEBAR_BOXED_RADIO_CSS = """
<style>
[data-testid="stSidebar"] [role="radiogroup"]{ display:flex; flex-direction:column; gap:8px; }
[data-testid="stSidebar"] [role="radiogroup"] > label{
  width:100%; display:flex; align-items:center; padding:10px 12px;
  border:1px solid rgba(255,255,255,0.12); border-radius:12px; background:rgba(255,255,255,0.03);
  transition: background .15s ease, border-color .15s ease, transform .05s ease; box-sizing:border-box;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover{ background:rgba(255,255,255,0.06); border-color:rgba(255,255,255,0.20); cursor:pointer; }
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child{ display:none !important; }
[data-testid="stSidebar"] [role="radiogroup"] > label > div:nth-child(2){ margin-left:0; }
[data-testid="stSidebar"] [role="radiogroup"] > label[aria-checked="true"]{
  background:linear-gradient(90deg, rgba(59,130,246,.20), rgba(59,130,246,.10));
  border-color:rgba(59,130,246,.45); box-shadow:0 0 0 1px rgba(59,130,246,.35) inset;
}
[data-testid="stSidebar"] [role="radiogroup"] > label svg{ margin-right:8px; }
</style>
"""
st.markdown(SIDEBAR_BOXED_RADIO_CSS, unsafe_allow_html=True)

def metric_card(title: str, value: str, color: str = "#4F46E5"):
    st.markdown(
        f"""
        <div class="kpi-card" style="background:{color};">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Helpers
# =========================
def wrap_labels(df, col, width=12):
    out = df.copy()
    out[col] = out[col].astype(str).apply(lambda s: "<br>".join(textwrap.wrap(s, width=width)) or s)
    return out

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def find_col(df: pd.DataFrame, aliases):
    m = { _norm(c): c for c in df.columns }
    for a in aliases:
        if _norm(a) in m:
            return m[_norm(a)]
    for c in df.columns:
        if any(_norm(a) in _norm(c) for a in aliases):
            return c
    return None

def clean_cat(s: pd.Series) -> pd.Series:
    z = s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    mapping = {
        "Hukum": "Ilmu Hukum",
        "hi": "Hubungan Internasional",
        "Hi": "Hubungan Internasional",
        "Tek Kimia": "Teknik Kimia",
    }
    z = z.replace(mapping)
    return z

@st.cache_data(show_spinner=False)
def read_csv_textarea(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text))

def try_scipy_pearsonr(x: pd.Series, y: pd.Series) -> Tuple[float, Optional[float]]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    d = pd.concat([x, y], axis=1).dropna()
    if d.empty:
        return (np.nan, None)
    r = float(np.corrcoef(d.iloc[:,0], d.iloc[:,1])[0,1])
    try:
        from scipy import stats
        _, p = stats.pearsonr(d.iloc[:,0], d.iloc[:,1])
        return (r, float(p))
    except Exception:
        return (r, None)

def try_scipy_linregress(x: pd.Series, y: pd.Series):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    d = pd.concat([x, y], axis=1).dropna()
    if d.empty:
        return {"slope": np.nan, "intercept": np.nan, "rvalue": np.nan, "pvalue": None}
    try:
        from scipy import stats
        res = stats.linregress(d.iloc[:,0], d.iloc[:,1])
        return {"slope": float(res.slope), "intercept": float(res.intercept),
                "rvalue": float(res.rvalue), "pvalue": float(res.pvalue)}
    except Exception:
        slope, intercept = np.polyfit(d.iloc[:,0], d.iloc[:,1], deg=1)
        r = float(np.corrcoef(d.iloc[:,0], d.iloc[:,1])[0,1])
        return {"slope": float(slope), "intercept": float(intercept),
                "rvalue": r, "pvalue": None}

def fmt_money(x):
    try:
        return f"Rp {int(x):,}".replace(",", ".")
    except Exception:
        return "–"

def safe_fmt(val, fmt="{:.2f}"):
    try:
        return fmt.format(float(val))
    except Exception:
        return "–"

def classify_r(r: float) -> str:
    if not np.isfinite(r): return "–"
    ar = abs(r)
    if ar < 0.1: return "sangat lemah"
    if ar < 0.3: return "lemah"
    if ar < 0.5: return "sedang"
    return "kuat"

def interpret_text(p, r, slope, lift, x_name, y_name) -> str:
    ptxt = "tidak dihitung" if p is None else f"{p:.3f}"
    dir_text = "positif" if (np.isfinite(slope) and slope > 0) else ("negatif" if np.isfinite(slope) and slope < 0 else "tidak jelas")
    r_class = classify_r(r) if np.isfinite(r) else "–"
    lift_dir = "lebih tinggi" if (np.isfinite(lift) and lift > 0) else ("lebih rendah" if np.isfinite(lift) and lift < 0 else "nyaris sama")
    if p is not None and p < 0.05:
        core = f"Ada bukti **hubungan linear** antara **{x_name}** dan **{y_name}** (p={ptxt})."
    else:
        core = f"**Belum ada bukti** yang cukup untuk hubungan linear antara **{x_name}** dan **{y_name}** (p={ptxt})."
    eff = f" Arah global cenderung **{dir_text}** (β₁={safe_fmt(slope,'{:.3f}')}); kekuatan korelasi **{r_class}** (r={safe_fmt(r,'{:.3f}')} )."
    lift_txt = f" Perbedaan rata-rata (Δ Q4−Q1) = **{safe_fmt(lift,'{:.3f}')}**, artinya outcome pada kuartil paparan tertinggi **{lift_dir}** dibanding kuartil terendah."
    extra = ""
    if (p is None or p >= 0.05):
        extra = " Pola bin-means yang datar/berfluktuasi kecil menandakan efek praktis kemungkinan **kecil**. Coba cek hubungan **non-linear** (LOWESS) atau korelasi **Spearman**."
    return core + eff + lift_txt + extra

def interpret_bubble(agg_df: pd.DataFrame, level: str, agg_name: str) -> str:
    if agg_df.empty:
        return "Tidak ada data untuk diinterpretasi."
    top_row = agg_df.loc[agg_df["avg_biaya"].idxmax()]
    bot_row = agg_df.loc[agg_df["avg_biaya"].idxmin()]
    big_row = agg_df.loc[agg_df["count"].idxmax()]
    rng = float(top_row["avg_biaya"] - bot_row["avg_biaya"])
    mean_val = float(agg_df["avg_biaya"].mean())
    cv = (np.std(agg_df["avg_biaya"], ddof=0) / mean_val) if mean_val else np.nan
    return (
        f"Pada agregasi **{agg_name}**, {level.lower()} **{top_row['kelompok']}** tertinggi (≈ {fmt_money(top_row['avg_biaya'])}), "
        f"terendah **{bot_row['kelompok']}** (≈ {fmt_money(bot_row['avg_biaya'])}). Responden terbanyak: **{big_row['kelompok']}** "
        f"({int(big_row['count'])} orang). Rentang ≈ **{fmt_money(rng)}**; CV ≈ **{safe_fmt(cv,'{:.2f}')}**."
    )

# ======== AUTO-LOAD CSV from local folder (tanpa upload) ========
DEFAULT_CSV_NAME = "data_cleaned_ver2(1).csv"

def read_csv_robust(path: str) -> Optional[pd.DataFrame]:
    for enc in [None, "utf-8", "utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc) if enc else pd.read_csv(path)
        except Exception:
            continue
    return None

def autoload_local_csv(preferred_name: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    candidates = []
    # jika user mengisi path absolut/nama file
    if preferred_name:
        if os.path.isabs(preferred_name):
            candidates.append(preferred_name)
        # relatif
        candidates += [
            preferred_name,
            os.path.join("data", preferred_name),
            os.path.join("dataset", preferred_name),
            os.path.join("datasets", preferred_name),
            os.path.join(os.getcwd(), preferred_name),
            os.path.join("/mnt/data", preferred_name),
        ]
    # explicit fallback yang sering dipakai
    candidates.append("/mnt/data/data_cleaned_ver2(1).csv")
    # cari secara rekursif berdasarkan nama file
    try:
        candidates += glob.glob(f"**/{preferred_name}", recursive=True)
    except Exception:
        pass

    seen = set()
    for p in candidates:
        p2 = os.path.normpath(p)
        if p2 in seen: 
            continue
        seen.add(p2)
        if os.path.isfile(p2):
            df = read_csv_robust(p2)
            if df is not None:
                return df, p2
    return None, None

# =========================
# Sidebar nav & data source
# =========================
with st.sidebar:
    st.title("🧭 Navigasi")
    page = st.radio(
        "Pilih halaman",
        [
            "📊 Beranda",
            "📈 Scatter",
            "📦 Distribusi",
            "🔥 Korelasi",
            "🧱 Komposisi Perangkat/Platform",
            "📑 Ringkasan Biaya",
            "🗂️ Data",
        ],
        index=0
    )
    st.markdown("---")
    st.subheader("📁 Sumber Data")
    data_mode = st.radio(
        "Mode input",
        ["Auto (CSV lokal)", "Upload CSV", "Paste CSV", "Input Manual (Editor)"],
        index=0
    )

    uploaded = None
    pasted_text = None
    auto_name = DEFAULT_CSV_NAME
    used_path_msg = ""
    if data_mode == "Upload CSV":
        uploaded = st.file_uploader("Unggah CSV", type=["csv"])
    elif data_mode == "Paste CSV":
        example = "Fakultas_norm,program studi_clean,biaya_internet_clean,Lama_Penggunaan_Jam,IPK\nFISIP,Ilmu Komunikasi,150000,2.0,3.2\nFH,Ilmu Hukum,200000,1.5,3.5\nFASILKOM,Sains Data,250000,2.5,3.7"
        pasted_text = st.text_area("Tempel CSV di sini", value=example, height=160)
        parse = st.button("Parse")
        st.markdown("---")
    elif data_mode == "Auto (CSV lokal)":
        auto_name = st.text_input("Nama/path CSV (otomatis)", value=DEFAULT_CSV_NAME,
                                  help="Isi dengan nama file atau path relatif/absolut. App akan mencari otomatis di beberapa folder umum.")

    st.caption("Gunakan halaman **Data** untuk unduh dataset hasil filter.")

if "manual_df" not in st.session_state:
    st.session_state.manual_df = pd.DataFrame({"Fakultas_norm": [], "program studi_clean": []})

# =========================
# Load data
# =========================
df = None
found_path = None
if data_mode == "Auto (CSV lokal)":
    df, found_path = autoload_local_csv(auto_name)
    if df is not None:
        st.sidebar.success(f"CSV terbaca otomatis: {found_path}")
    else:
        st.sidebar.error("Gagal membaca CSV secara otomatis. Periksa nama/path file, atau gunakan mode lain.")
elif data_mode == "Upload CSV" and uploaded is not None:
    df = pd.read_csv(uploaded)
elif data_mode == "Paste CSV" and pasted_text and 'parse' in locals() and parse:
    try:
        df = read_csv_textarea(pasted_text)
        st.sidebar.success("CSV berhasil dibaca.")
    except Exception as e:
        st.sidebar.error(f"Gagal parse CSV: {e}")
elif data_mode == "Input Manual (Editor)":
    st.markdown("### ✍️ Input Data Manual")
    st.caption("Minimal butuh dua kolom: **Fakultas_norm** dan **program studi_clean** (boleh nama lain—app akan deteksi).")
    edited = st.data_editor(
        st.session_state.manual_df, num_rows="dynamic", use_container_width=True,
        height=320, key="editor"
    )
    if st.button("Simpan Data Manual"):
        st.session_state.manual_df = edited.copy()
        st.success("Data manual disimpan.")
    if not edited.empty:
        df = edited.copy()

if df is None or df.empty:
    st.info("Belum ada data. Aktifkan mode **Auto (CSV lokal)** dengan nama/path yang benar, atau gunakan Upload/Paste/Editor.")
    st.stop()

# =========================
# Detect columns
# =========================
FAK_ALIASES   = ["Fakultas_norm", "Fakultas", "Fakultas(Jangan Disingkat)", "Faculty"]
PRODI_ALIASES = ["program studi_clean", "Program Studi", "Prodi", "Program_Studi", "Program Studi Clean"]
BIAYA_ALIASES = [
    "biaya_internet_clean", "Biaya_internet_clean", "biaya_internet",
    "Biaya Internet", "biaya", "pengeluaran_internet", "biaya per bulan"
]
DEVICE_ALIASES   = ["Perangkat_yang_sering_digunakan","Perangkat","Device"]
PLATFORM_ALIASES = ["platform/aplikasi_untuk_pembelajaran_online","Platform","Aplikasi Platform"]
JAM_ALIASES      = ["Lama_Penggunaan_Jam", "Lama Penggunaan Jam", "lama_penggunaan_jam"]
JAM_LOG_ALIASES  = ["Lama_Penggunaan_Jam_log"]

fak_col      = find_col(df, FAK_ALIASES)
prodi_col    = find_col(df, PRODI_ALIASES)
biaya_col    = find_col(df, BIAYA_ALIASES)
device_col   = find_col(df, DEVICE_ALIASES)
platform_col = find_col(df, PLATFORM_ALIASES)
jam_col      = find_col(df, JAM_ALIASES)
jam_log_col  = find_col(df, JAM_LOG_ALIASES)

if fak_col is None or prodi_col is None:
    miss = []
    if fak_col is None: miss.append("Fakultas")
    if prodi_col is None: miss.append("Program Studi")
    st.error(f"Kolom tidak ditemukan: {', '.join(miss)}. Kolom tersedia: {list(df.columns)}")
    st.stop()

df[fak_col] = clean_cat(df[fak_col])
df[prodi_col] = clean_cat(df[prodi_col])

# =========================
# GLOBAL PAGE TITLE
# =========================
st.markdown("""
<div style="display:flex;align-items:center;gap:.6rem;margin:6px 0 14px 0;">
  <span style="font-size:1.9rem;font-weight:800;letter-spacing:.2px;">
    Pengaruh Teknologi terhadap Proses Belajar Mahasiswa
  </span>
</div>
""", unsafe_allow_html=True)

# =========================
# Top filter bar
# =========================
fak_opts   = ["All"] + sorted(df[fak_col].dropna().unique().tolist())
prodi_opts = ["All"] + sorted(df[prodi_col].dropna().unique().tolist())

f1, f2 = st.columns(2)
with f1:
    sel_fak = st.selectbox("🎓 Fakultas (Filter)", fak_opts, index=0)
with f2:
    sel_pro = st.selectbox("📚 Program Studi (Filter)", prodi_opts, index=0)

filtered = df.copy()
if sel_fak != "All":
    filtered = filtered[filtered[fak_col] == sel_fak]
if sel_pro != "All":
    filtered = filtered[filtered[prodi_col] == sel_pro]

num_cols = sorted([c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])])
cat_cols = sorted([c for c in filtered.columns if (filtered[c].dtype == "object" or pd.api.types.is_categorical_dtype(filtered[c]))])

# =========================
# Common charts
# =========================
def chart_count(data: pd.DataFrame, cat_col: str, title: str, force_bar: bool = True):
    if data.empty:
        st.warning("Tidak ada data untuk divisualisasikan.")
        return
    counts = data[cat_col].value_counts(dropna=False).reset_index()
    counts.columns = ["Kategori", "Jumlah"]
    if force_bar:
        counts_wrapped = wrap_labels(counts, "Kategori", width=14)
        fig = px.bar(counts_wrapped, x="Kategori", y="Jumlah", color="Kategori",
                     title=title, text="Jumlah")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="", yaxis_title="Jumlah", legend_title="")
    else:
        fig = px.pie(counts, names="Kategori", values="Jumlah", hole=0.35, title=title)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(legend_title="")
    fig.update_layout(modebar_add=["toImage"])
    st.plotly_chart(fig, use_container_width=True)

def scatter_cat_num(data: pd.DataFrame, cat_col: str, num_col: str, title: str, color_by: Optional[str] = None):
    if data.empty:
        st.warning("Tidak ada data untuk divisualisasikan.")
        return
    if (cat_col not in data.columns) or (num_col not in data.columns):
        st.warning("Kolom tidak ditemukan di data terfilter.")
        return
    d = data[[cat_col, num_col] + ([color_by] if color_by else [])].dropna()
    if d.empty:
        st.info("Semua nilai kosong setelah filter. Coba ubah filter atau kolom.")
        return
    fig = px.scatter(d, x=cat_col, y=num_col, color=color_by,
                     hover_data=d.columns, title=title, opacity=0.75)
    fig.update_layout(xaxis_title=cat_col, yaxis_title=num_col, legend_title="", modebar_add=["toImage"])
    st.plotly_chart(fig, use_container_width=True)

def flag_outliers(df_in, x, y, method="IQR", z=3.0):
    df = df_in.copy()
    if df.empty: return pd.Series(False, index=df.index)
    if method.upper() == "IQR":
        xq1, xq3 = np.percentile(df[x], [25, 75])
        yq1, yq3 = np.percentile(df[y], [25, 75])
        xi, yi = (xq3 - xq1), (yq3 - yq1)
        xlow, xhigh = xq1 - 1.5 * xi, xq3 + 1.5 * xi
        ylow, yhigh = yq1 - 1.5 * yi, yq3 + 1.5 * yi
        return (df[x] < xlow) | (df[x] > xhigh) | (df[y] < ylow) | (df[y] > yhigh)
    else:
        zx = (df[x] - df[x].mean()) / (df[x].std(ddof=0) or 1)
        zy = (df[y] - df[y].mean()) / (df[y].std(ddof=0) or 1)
        return (zx.abs() > z) | (zy.abs() > z)

# =========================
# Pages
# =========================
if page == "📊 Beranda":
    st.subheader("📊 Ringkasan")
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Jumlah Responden", f"{len(filtered):,}", "#2563EB")
    with c2: metric_card("Variabel Fakultas (unique)", f"{filtered[fak_col].nunique():,}", "#059669")
    with c3: metric_card("Variabel Program Studi (unique)", f"{filtered[prodi_col].nunique():,}", "#DC2626")

    st.markdown("### ⚡ KPI Pengaruh Teknologi")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    jam_col_use = jam_log_col if (jam_log_col in filtered.columns) else (find_col(filtered, ["Lama_Penggunaan_Jam"]) or jam_col)

    with c1:
        if jam_col_use and jam_col_use in filtered:
            st.caption("Paparan")
            metric_card("Rata-rata Jam/Hari", safe_fmt(filtered[jam_col_use].mean(), "{:.2f}"), "#0ea5e9")
        else:
            metric_card("Rata-rata Jam/Hari", "–", "#0ea5e9")

    with c2:
        st.caption("Akses")
        if biaya_col and biaya_col in filtered:
            metric_card("Median Biaya Internet", fmt_money(filtered[biaya_col].median()), "#10b981")
        else:
            metric_card("Median Biaya Internet", "–", "#10b981")

    freq_col = find_col(filtered, ["frekuensi_penggunaan_platform", "frekuensi platform", "freq_platform"])
    with c3:
        st.caption("Intensitas")
        if freq_col and freq_col in filtered and pd.api.types.is_numeric_dtype(filtered[freq_col]):
            metric_card("Median Frekuensi/minggu", safe_fmt(filtered[freq_col].median(), "{:.0f}"), "#f59e0b")
        else:
            metric_card("Median Frekuensi/minggu", "–", "#f59e0b")

    with c4:
        st.caption("Perangkat")
        if device_col and device_col in filtered:
            is_laptop = filtered[device_col].astype(str).str.contains("laptop", case=False, na=False)
            metric_card("% Pakai Laptop", f"{is_laptop.mean():.0%}", "#a855f7")
        else:
            metric_card("% Pakai Laptop", "–", "#a855f7")

    with c5:
        st.caption("Platform")
        if platform_col and platform_col in filtered and not filtered[platform_col].dropna().empty:
            top_plat = filtered[platform_col].value_counts().idxmax()
            metric_card("Top Platform", str(top_plat), "#ef4444")
        else:
            metric_card("Top Platform", "–", "#ef4444")

    # ====== FIX: KPI Korelasi otomatis pilih pasangan terbaik ======
    with c6:
        st.caption("Korelasi Teknologi↔Hasil")
        OUTCOME_ALIASES = ["prestasi", "nilai", "IPK", "motivasi_belajar", "motivasi", "engagement"]
        out_col = find_col(filtered, OUTCOME_ALIASES)
        # fallback: ambil kolom numerik lain selain kandidat X
        candidates = []
        if jam_col_use and jam_col_use in filtered:
            candidates.append(("Jam/Hari", jam_col_use))
        if biaya_col and biaya_col in filtered:
            candidates.append(("Biaya Internet", biaya_col))
        if freq_col and freq_col in filtered and pd.api.types.is_numeric_dtype(filtered[freq_col]):
            candidates.append(("Frekuensi", freq_col))

        if out_col is None and num_cols:
            for c in num_cols:
                if c not in [col for _, col in candidates]:
                    out_col = c
                    break

        shown = False
        if candidates and out_col:
            for label, xcol in candidates:
                x = pd.to_numeric(filtered[xcol], errors="coerce")
                y = pd.to_numeric(filtered[out_col], errors="coerce")
                dtmp = pd.concat([x, y], axis=1).dropna()
                if len(dtmp) >= 3:
                    r, p = try_scipy_pearsonr(dtmp.iloc[:, 0], dtmp.iloc[:, 1])
                    star = "★" if (p is not None and p < 0.05) else ""
                    ptxt = f"p={p:.3f}" if p is not None else "p=–"
                    metric_card(f"r({xcol}↔{out_col})", f"{safe_fmt(r,'{:.2f}')} ({ptxt}) {star}", "#22c55e")
                    shown = True
                    break

        if not shown:
            metric_card("Korelasi Teknologi↔Hasil", "Data tidak cukup", "#22c55e")

    # ==== Visualisasi Utama (di atas Bubble) ====
    st.markdown("## Visualisasi Utama")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Distribusi Fakultas**")
        chart_count(filtered, fak_col, "Responden per Fakultas", force_bar=True)
    with v2:
        st.markdown("**Distribusi Program Studi**")
        chart_count(filtered, prodi_col, "Responden per Program Studi", force_bar=True)

    # ===== 💸 KPI Ringkasan Biaya (di atas bubble) =====
    st.markdown("### 💸 KPI Ringkasan Biaya Internet")
    if biaya_col:
        agg_choice = st.radio("Agregasi KPI", ["Rata-rata (mean)", "Median"], horizontal=True, key="agg_home_radio")
        aggfunc = "mean" if agg_choice.startswith("Rata") else "median"

        grp = filtered[[fak_col, prodi_col, biaya_col]].dropna()
        if not grp.empty:
            global_val = getattr(grp[biaya_col], aggfunc)()
            fac_tbl = grp.groupby(fak_col)[biaya_col].agg(aggfunc).sort_values(ascending=False)
            pro_tbl = grp.groupby(prodi_col)[biaya_col].agg(aggfunc).sort_values(ascending=False)
            top_fac_name = fac_tbl.index[0] if not fac_tbl.empty else "–"
            top_fac_val  = fac_tbl.iloc[0] if not fac_tbl.empty else np.nan
            top_pro_name = pro_tbl.index[0] if not pro_tbl.empty else "–"
            top_pro_val  = pro_tbl.iloc[0] if not pro_tbl.empty else np.nan

            k1, k2, k3 = st.columns(3)
            with k1: metric_card(f"Biaya Internet ({'Mean' if aggfunc=='mean' else 'Median'})", fmt_money(global_val), "#0ea5e9")
            with k2: metric_card(f"Fakultas Tertinggi ({aggfunc.title()})", f"{top_fac_name}<br>{fmt_money(top_fac_val)}", "#8b5cf6")
            with k3: metric_card(f"Prodi Tertinggi ({aggfunc.title()})", f"{top_pro_name}<br>{fmt_money(top_pro_val)}", "#f43f5e")
    else:
        st.info("Kolom biaya internet belum terdeteksi, KPI biaya tidak ditampilkan.")

    # ===== 🧩 BUBBLE CHART: Klaster =====
    st.markdown("### 🧩 Klaster Rata-rata Harga Kuota (Bubble Chart)")
    if biaya_col:
        level = st.selectbox("Kelompok", ["Fakultas", "Program Studi"], index=0, key="cluster_level")
        agg_choice2 = st.radio("Agregasi rata-rata kuota", ["Rata-rata (mean)", "Median"],
                               horizontal=True, key="cluster_agg")
        aggfunc2 = "mean" if agg_choice2.startswith("Rata") else "median"
        k = st.slider("Jumlah Klaster (K)", min_value=2, max_value=6, value=3, step=1, key="cluster_k")

        grp_col = fak_col if level == "Fakultas" else prodi_col
        g2 = filtered[[grp_col, biaya_col]].dropna()
        if g2.empty:
            st.info("Data biaya kosong setelah filter.")
        else:
            agg_df = g2.groupby(grp_col).agg(
                avg_biaya=(biaya_col, aggfunc2),
                count=(biaya_col, "size")
            ).reset_index().rename(columns={grp_col: "kelompok"})

            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
                Xs = StandardScaler().fit_transform(agg_df[["avg_biaya", "count"]].values)
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                agg_df["cluster"] = km.fit_predict(Xs)
            except Exception:
                # SAFE fallback tanpa mismatch labels saat qcut melakukan 'duplicates=drop'
                rank_vals = agg_df["avg_biaya"].rank(method="first")
                q_eff = min(k, rank_vals.nunique())
                if q_eff < 2:
                    agg_df["cluster"] = 0
                else:
                    cuts = pd.qcut(rank_vals, q=q_eff, duplicates="drop")
                    agg_df["cluster"] = pd.factorize(cuts)[0]  # 0..q_eff-1

            agg_df["label"] = agg_df["kelompok"]

            figc = px.scatter(
                agg_df, x="avg_biaya", y="count",
                color="cluster", size="count", size_max=60,
                hover_data=["kelompok", "avg_biaya", "count", "cluster"],
                text="label",
                title=f"Klaster {level}: ({aggfunc2.title()} biaya vs jumlah responden)"
            )
            figc.update_traces(textposition="top center", opacity=0.9)
            figc.update_layout(
                modebar_add=["toImage"],
                xaxis_title=f"{aggfunc2.title()} Biaya Internet",
                yaxis_title="Jumlah Responden",
                legend_title="Cluster"
            )
            st.plotly_chart(figc, use_container_width=True)

            with st.expander("Ringkasan Klaster"):
                summ = (agg_df.groupby("cluster")
                            .agg(jumlah_kelompok=("kelompok","count"),
                                 rata2_biaya=("avg_biaya","mean"),
                                 min_biaya=("avg_biaya","min"),
                                 max_biaya=("avg_biaya","max"),
                                 total_responden=("count","sum"))
                            .sort_index())
                st.dataframe(summ, use_container_width=True, height=260)

            st.markdown("#### 📝 Interpretasi Otomatis (Bubble)")
            st.write(interpret_bubble(agg_df, level, aggfunc2.title()))

    # ===== Pengaruh Teknologi Ringkas =====
    st.markdown("### 🧠 Pengaruh Teknologi (Ringkas)")
    num_only = filtered.select_dtypes(include="number")
    if not num_only.empty and num_only.shape[1] >= 2:
        exposure_default = (jam_col if (jam_col in num_only.columns) else num_only.columns[0])
        outcome_default  = find_col(filtered, ["prestasi","motivasi_belajar"]) or (num_only.columns[1] if num_only.shape[1]>1 else num_only.columns[0])

        cI1, cI2, cI3 = st.columns(3)
        with cI1:
            x_expo = st.selectbox("Exposure (X) – variabel teknologi", options=num_only.columns.tolist(),
                                  index=list(num_only.columns).index(exposure_default) if exposure_default in num_only.columns else 0, key="ix")
        with cI2:
            y_out  = st.selectbox("Outcome (Y) – hasil belajar", options=num_only.columns.tolist(),
                                  index=list(num_only.columns).index(outcome_default) if outcome_default in num_only.columns else (1 if num_only.shape[1]>1 else 0),
                                  key="iy")
        with cI3:
            color_by = st.selectbox("Warna berdasarkan", options=["(tanpa)", fak_col, prodi_col], index=1, key="icol")
            color_by = None if color_by == "(tanpa)" else color_by

        d_imp = filtered[[x_expo, y_out] + ([color_by] if color_by else [])].dropna()
        if not d_imp.empty:
            lr = try_scipy_linregress(d_imp[x_expo], d_imp[y_out])

            # ---------- SAFE QUARTILING (hindari ValueError labels vs edges) ----------
            vals = pd.to_numeric(d_imp[x_expo], errors="coerce")
            nuniq = vals.nunique(dropna=True)
            labels_map = {
                2: ["Q1 (rendah)", "Q2 (tinggi)"],
                3: ["Q1 (rendah)", "Q2", "Q3 (tinggi)"],
                4: ["Q1 (rendah)", "Q2", "Q3", "Q4 (tinggi)"],
            }
            q_eff = min(4, nuniq)  # kuartil efektif 2..4 sesuai variasi data

            d_lift = d_imp.copy()
            lift = np.nan
            if q_eff >= 2:
                bins = pd.qcut(vals.rank(method="first"), q=q_eff, labels=labels_map[q_eff], duplicates="drop")
                d_lift["quartile"] = bins
                low_label  = labels_map[q_eff][0]
                high_label = labels_map[q_eff][-1]
                q_high = d_lift.loc[d_lift["quartile"] == high_label, y_out].mean()
                q_low  = d_lift.loc[d_lift["quartile"] == low_label,  y_out].mean()
                lift = q_high - q_low
            else:
                d_lift["quartile"] = np.nan
            # -------------------------------------------------------------------------

            k1,k2,k3,k4 = st.columns(4)
            with k1: st.metric("Slope (β₁)", safe_fmt(lr["slope"], "{:.3f}"))
            with k2: st.metric("Pearson r", safe_fmt(lr["rvalue"], "{:.3f}"))
            with k3: st.metric("p-value", "—" if lr["pvalue"] is None else f"{lr['pvalue']:.3f}")
            with k4: st.metric("Δ (Q4−Q1)", safe_fmt(lift, "{:.3f}"))

            L, R = st.columns(2)
            with L:
                fig_sc = px.scatter(d_imp, x=x_expo, y=y_out, color=color_by, opacity=0.85,
                                    trendline="ols", title=f"{x_expo} → {y_out}")
                fig_sc.update_layout(legend_title="", modebar_add=["toImage"])
                st.plotly_chart(fig_sc, use_container_width=True)
            with R:
                nb = st.slider("Jumlah bin (rata-rata per bin X)", 5, 20, 10, key="bins_home")

                # ---------- Guard untuk BIN MEANS ----------
                d_bins = d_imp.copy()
                if d_bins[x_expo].nunique(dropna=True) < 2:
                    st.info("Variabel X belum cukup bervariasi untuk dibagi menjadi beberapa bin.")
                else:
                    nb_eff = min(nb, d_bins[x_expo].nunique(dropna=True))
                    d_bins["bin"] = pd.qcut(
                        pd.to_numeric(d_bins[x_expo], errors="coerce").rank(method="first"),
                        nb_eff,
                        duplicates="drop"
                    )
                    if d_bins["bin"].notna().sum() >= nb_eff:
                        agg = (d_bins.groupby("bin")
                                     .agg(x_mid=(x_expo,"mean"), y_mean=(y_out,"mean"), n=("bin","size"))
                                     .reset_index())
                        fig_bin = px.line(agg, x="x_mid", y="y_mean", markers=True, text="n",
                                          title=f"Rata-rata {y_out} per bin {x_expo}")
                        fig_bin.update_traces(textposition="top center")
                        fig_bin.update_layout(modebar_add=["toImage"], xaxis_title=x_expo, yaxis_title=y_out)
                        st.plotly_chart(fig_bin, use_container_width=True)
                # ------------------------------------------

            st.markdown("#### 📝 Interpretasi Otomatis")
            st.write(interpret_text(lr["pvalue"], lr["rvalue"], lr["slope"], lift, x_expo, y_out))

    st.markdown("### 🔥 Matriks Korelasi (Ringkas)")
    if not num_only.empty and not num_only.corr(numeric_only=True).empty:
        fig = px.imshow(num_only.corr(numeric_only=True), text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r", origin="lower", title="Matriks Korelasi")
        fig.update_layout(modebar_add=["toImage"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cuplikan Data (scrollable)")
    st.dataframe(filtered.head(100), use_container_width=True, height=320)

elif page == "📈 Scatter":
    st.subheader("📈 Scatter")
    if len(num_cols) >= 2:
        c1, c2, c3 = st.columns(3)
        with c1:
            x_num = st.selectbox("Kolom X (numerik)", options=num_cols, index=0, key="numx")
        with c2:
            y_num = st.selectbox("Kolom Y (numerik)", options=num_cols, index=(1 if len(num_cols)>1 else 0), key="numy")
        with c3:
            color_by2 = st.selectbox("Warna berdasarkan", [fak_col, prodi_col, "(tanpa)"], index=0, key="numcolor")
            color_by2 = None if color_by2 == "(tanpa)" else color_by2

        with st.expander("Opsi lanjutan (opsional)", expanded=False):
            trend = st.selectbox("Trendline", ["(none)", "ols", "lowess"], index=1)
            trend = None if trend == "(none)" else trend
            outlier_method = st.selectbox("Highlight outlier dengan", ["(tanpa)", "IQR", "Z-score"], index=0)

        keep_cols = [x_num, y_num] + ([color_by2] if color_by2 else [])
        dn = filtered[keep_cols].dropna().copy()

        symbol_col = None; symbol_map = None
        if outlier_method != "(tanpa)":
            dn["Outlier"] = flag_outliers(dn, x_num, y_num, method=("IQR" if outlier_method=="IQR" else "Z"), z=3.0)
            symbol_col = "Outlier"; symbol_map = {False: "circle", True: "x"}

        if not dn.empty:
            fig_nn = px.scatter(dn, x=x_num, y=y_num, color=color_by2, symbol=symbol_col, symbol_map=symbol_map,
                                opacity=0.85, hover_data=dn.columns, trendline=trend, title=f"{x_num} vs {y_num}")
            fig_nn.update_layout(legend_title=str(color_by2) if color_by2 else "", modebar_add=["toImage"])
            st.plotly_chart(fig_nn, use_container_width=True)
    st.markdown("---")

    st.subheader("🧭 Scatter Explorer (Kategorikal vs Numerik)")
    if cat_cols and num_cols:
        e1, e2, e3 = st.columns(3)
        with e1:
            default_cat_idx = cat_cols.index(prodi_col) if prodi_col in cat_cols else 0
            cat_pick = st.selectbox("Kolom Kategorikal (X)", options=cat_cols, index=default_cat_idx)
        with e2:
            default_num_idx = num_cols.index(biaya_col) if (biaya_col in num_cols) else 0
            num_pick = st.selectbox("Kolom Numerik (Y)", options=num_cols, index=default_num_idx if num_cols else 0)
        with e3:
            color_pick = st.selectbox("Warna berdasarkan (opsional)", options=["(tanpa)"] + cat_cols, index=0)
            color_pick = None if color_pick == "(tanpa)" else color_pick
        scatter_cat_num(filtered, cat_pick, num_pick, f"{cat_pick} vs {num_pick}", color_by=color_pick)

elif page == "📦 Distribusi":
    st.subheader("📦 Distribusi per Kategori")
    if not num_cols or not cat_cols:
        st.warning("Butuh minimal 1 kolom numerik dan 1 kolom kategorikal.")
    else:
        ctop1, ctop2 = st.columns(2)
        with ctop1:
            idx_num = num_cols.index(biaya_col) if (biaya_col in num_cols) else 0
            y_num = st.selectbox("Kolom Numerik (Y)", options=num_cols, index=idx_num)
        with ctop2:
            idx_cat = cat_cols.index(fak_col) if fak_col in cat_cols else 0
            cat_x = st.selectbox("Kelompok (X)", options=cat_cols, index=idx_cat)

        st.markdown(f"**Box Plot: {cat_x} vs {y_num}**")
        d = filtered[[cat_x, y_num]].dropna()
        if not d.empty:
            fig = px.box(d, x=cat_x, y=y_num, points="outliers", title=f"{cat_x} vs {y_num}")
            fig.update_layout(modebar_add=["toImage"])
            st.plotly_chart(fig, use_container_width=True)

elif page == "🔥 Korelasi":
    st.subheader("🔥 Korelasi Antar Variabel Numerik")
    num_only = filtered.select_dtypes(include="number")
    if not num_only.empty and num_only.shape[1] >= 2:
        fig = px.imshow(num_only.corr(numeric_only=True), text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r", origin="lower", title="Matriks Korelasi")
        fig.update_layout(modebar_add=["toImage"])
        st.plotly_chart(fig, use_container_width=True)

elif page == "🧱 Komposisi Perangkat/Platform":
    st.subheader("🧱 Komposisi Perangkat/Platform per Fakultas")

    c1, c2 = st.columns(2)
    with c1:
        if device_col and device_col in filtered.columns:
            d = filtered[[fak_col, device_col]].dropna().copy()
            if d.empty:
                st.info("Data perangkat kosong.")
            else:
                d["Count"] = 1
                pv = d.pivot_table(index=fak_col, columns=device_col, values="Count", aggfunc="sum", fill_value=0)
                fig = px.bar(pv, x=pv.index, y=pv.columns, title="Perangkat per Fakultas", barmode="stack")
                fig.update_layout(xaxis_title=fak_col, yaxis_title="Jumlah", legend_title=str(device_col), modebar_add=["toImage"])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Kolom perangkat tidak ditemukan.")

    with c2:
        if platform_col and platform_col in filtered.columns:
            d2 = filtered[[fak_col, platform_col]].dropna().copy()
            if d2.empty:
                st.info("Data platform kosong.")
            else:
                d2["Count"] = 1
                pv2 = d2.pivot_table(index=fak_col, columns=platform_col, values="Count", aggfunc="sum", fill_value=0)
                fig2 = px.bar(pv2, x=pv2.index, y=pv2.columns, title="Platform per Fakultas", barmode="stack")
                fig2.update_layout(xaxis_title=fak_col, yaxis_title="Jumlah", legend_title=str(platform_col), modebar_add=["toImage"])
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Kolom platform tidak ditemukan.")

    st.markdown("---")
    st.subheader("🏆 Top Platform & Perangkat per Prodi (proporsi)")

    # ----- Top Platform per Prodi -----
    if platform_col and platform_col in filtered.columns:
        dp = filtered[[prodi_col, platform_col]].dropna()
        if not dp.empty:
            tot = dp.groupby(prodi_col).size().rename("total")
            top = (dp.groupby([prodi_col, platform_col]).size()
                    .rename("cnt").reset_index())
            top_idx = top.groupby(prodi_col)["cnt"].idxmax()
            top_pl = top.loc[top_idx].copy().merge(tot, on=prodi_col)
            top_pl["share"] = top_pl["cnt"] / top_pl["total"]
            top_pl["label"] = (top_pl[platform_col].astype(str)
                               + " (" + top_pl["cnt"].astype(str) + "/" + top_pl["total"].astype(str) + ")")
            top_pl["_y"] = top_pl[prodi_col].astype(str).apply(lambda s: "<br>".join(textwrap.wrap(s, 24)))

            h = min(1400, max(450, 32 * top_pl.shape[0] + 160))

            figp = px.bar(
                top_pl.sort_values("share", ascending=False),
                x="share", y="_y", color=platform_col, text="label",
                orientation="h", title="Top Platform per Prodi"
            )
            figp.update_traces(textposition="outside", cliponaxis=False, textfont_size=12)
            figp.update_layout(
                xaxis_tickformat=".0%", legend_title=str(platform_col),
                modebar_add=["toImage"], height=h,
                margin=dict(l=220, r=40, t=60, b=40)
            )
            st.plotly_chart(figp, use_container_width=True)
        else:
            st.info("Tidak ada data platform setelah filter.")

    # ----- Top Perangkat per Prodi -----
    if device_col and device_col in filtered.columns:
        dd = filtered[[prodi_col, device_col]].dropna()
        if not dd.empty:
            tot2 = dd.groupby(prodi_col).size().rename("total")
            top2 = (dd.groupby([prodi_col, device_col]).size()
                     .rename("cnt").reset_index())
            top_idx2 = top2.groupby(prodi_col)["cnt"].idxmax()
            top_dev = top2.loc[top_idx2].copy().merge(tot2, on=prodi_col)
            top_dev["share"] = top_dev["cnt"] / top_dev["total"]
            top_dev["label"] = (top_dev[device_col].astype(str)
                                + " (" + top_dev["cnt"].astype(str) + "/" + top_dev["total"].astype(str) + ")")
            top_dev["_y"] = top_dev[prodi_col].astype(str).apply(lambda s: "<br>".join(textwrap.wrap(s, 24)))

            h2 = min(1400, max(450, 32 * top_dev.shape[0] + 160))

            figd = px.bar(
                top_dev.sort_values("share", ascending=False),
                x="share", y="_y", color=device_col, text="label",
                orientation="h", title="Top Perangkat per Prodi"
            )
            figd.update_traces(textposition="outside", cliponaxis=False, textfont_size=12)
            figd.update_layout(
                xaxis_tickformat=".0%", legend_title=str(device_col),
                modebar_add=["toImage"], height=h2,
                margin=dict(l=220, r=40, t=60, b=40)
            )
            st.plotly_chart(figd, use_container_width=True)
        else:
            st.info("Tidak ada data perangkat setelah filter.")

elif page == "📑 Ringkasan Biaya":
    st.subheader("📑 Ringkasan Biaya Internet")
    if biaya_col:
        mode = st.radio("Agregasi", ["Rata-rata (mean)", "Median"], horizontal=True)
        aggfunc = "mean" if mode.startswith("Rata") else "median"
        c1, c2 = st.columns(2)
        with c1:
            df_fac = filtered[[fak_col, biaya_col]].dropna().groupby(fak_col)[biaya_col].agg(aggfunc).reset_index()
            df_fac = df_fac.sort_values(by=biaya_col, ascending=False)
            st.markdown("**Fakultas**")
            st.dataframe(df_fac, use_container_width=True, height=320)
            fig = px.bar(df_fac, x=fak_col, y=biaya_col, title=f"{aggfunc.title()} Biaya per Fakultas")
            fig.update_layout(modebar_add=["toImage"])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            df_pro = filtered[[prodi_col, biaya_col]].dropna().groupby(prodi_col)[biaya_col].agg(aggfunc).reset_index()
            df_pro = df_pro.sort_values(by=biaya_col, ascending=False)
            st.markdown("**Program Studi**")
            st.dataframe(df_pro, use_container_width=True, height=320)
            fig2 = px.bar(df_pro, x=prodi_col, y=biaya_col, title=f"{aggfunc.title()} Biaya per Prodi")
            fig2.update_layout(modebar_add=["toImage"])
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Kolom biaya internet belum terdeteksi.")

elif page == "🗂️ Data":
    st.subheader("🗂️ Data Lengkap")
    st.dataframe(filtered, use_container_width=True, height=560)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh Data (CSV)", data=csv, file_name="data_filtered.csv", mime="text/csv")

# python -m streamlit run app.py

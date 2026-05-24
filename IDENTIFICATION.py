

import numpy as np
import cv2
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import time

# ═══════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════

DENOMS = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
NAMES  = {10_000:"10.000 VNĐ", 20_000:"20.000 VNĐ", 50_000:"50.000 VNĐ",
          100_000:"100.000 VNĐ", 200_000:"200.000 VNĐ", 500_000:"500.000 VNĐ"}
COLORS = {10_000:"#B5651D", 20_000:"#1A6EB5", 50_000:"#C0392B",
          100_000:"#27AE60", 200_000:"#E67E22", 500_000:"#16A085"}
EMOJI  = {10_000:"🟤", 20_000:"🔵", 50_000:"🔴",
          100_000:"🟢", 200_000:"🟠", 500_000:"🟦"}
DESC   = {10_000: ("Nâu cam",   "Văn Miếu"),
          20_000: ("Xanh lam",  "Chùa Cầu Hội An"),
          50_000: ("Đỏ hồng",   "Kinh thành Huế"),
          100_000:("Xanh lá",   "Văn Miếu Quốc Tử Giám"),
          200_000:("Cam đỏ",    "Vịnh Hạ Long"),
          500_000:("Xanh teal", "Hồ Hoàn Kiếm")}


H_BINS, S_BINS, V_BINS = 18, 8, 8
N_FEAT = 56

def extract_features(img_bgr):

    img = cv2.resize(img_bgr, (160, 80))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    H = hsv[:,:,0]/180.0
    S = hsv[:,:,1]/255.0
    V = hsv[:,:,2]/255.0

    valid = (V > 0.08) & (V < 0.97)
    n = valid.sum() + 1e-9

    Hv = H[valid]
    Sv = S[valid]
    Vv = V[valid]
    if Hv.size == 0:
       Hv = np.array([0.0], dtype=np.float32)
       Sv = np.array([0.0], dtype=np.float32)
       Vv = np.array([0.0], dtype=np.float32)

    h_hist = np.histogram(Hv, bins=H_BINS, range=(0,1))[0]/n
    s_hist = np.histogram(Sv, bins=S_BINS, range=(0,1))[0]/n
    v_hist = np.histogram(Vv, bins=V_BINS, range=(0,1))[0]/n

    # màu
    red    = ((Hv<0.05)|(Hv>0.94)).sum()/n
    orange = ((Hv>=0.04)&(Hv<0.11)).sum()/n
    yellow = ((Hv>=0.10)&(Hv<0.17)).sum()/n
    green  = ((Hv>=0.20)&(Hv<0.42)).sum()/n
    teal   = ((Hv>=0.42)&(Hv<0.55)).sum()/n
    blue   = ((Hv>=0.55)&(Hv<0.72)).sum()/n

    # edge feature
    edges = cv2.Canny(gray, 80, 150)
    edge_density = edges.mean()/255.0

    # texture feature
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    texture = lap.var()/1000.0
    # ORB keypoints
    orb = cv2.ORB_create(nfeatures=120)

    kp = orb.detect(gray, None)
    kp_count = len(kp)

    if kp_count > 0:
        responses = [k.response for k in kp]
        mean_resp = np.mean(responses)
    else:
        mean_resp = 0.0

    orb_density = kp_count / gray.size

    # brightness regions
    dark = (gray < 70).mean()
    mid  = ((gray>=70)&(gray<170)).mean()
    bright = (gray>=170).mean()

    extra = np.array([
    red,orange,yellow,green,teal,blue,

    Hv.mean(),
    Sv.mean(),
    Vv.mean(),

    edge_density,
    texture,

    orb_density,
    mean_resp,

    dark,
    mid,
    bright
], dtype=np.float32)

    feat = np.concatenate([
        h_hist,
        s_hist,
        v_hist,
        extra
    ]).astype(np.float32)

    return feat

# ═══════════════════════════════════════════════════════
#  COLOR PROFILES
# ═══════════════════════════════════════════════════════

_PROFILES = [

    # 10k
    dict(h=0.075,hs=0.025,s=0.50,ss=0.12,
         v=0.63,vs=0.10,sh=0.18,sr=0.12),

    # 20k
    dict(h=0.60,hs=0.035,s=0.72,ss=0.09,
         v=0.62,vs=0.09,sh=0.50,sr=0.22),

    # 50k
    dict(h=0.96,hs=0.028,s=0.60,ss=0.10,
         v=0.72,vs=0.09,sh=0.10,sr=0.18),

    # 100k
    dict(h=0.32,hs=0.025,s=0.52,ss=0.09,
         v=0.68,vs=0.08,sh=0.22,sr=0.20),

    # 200k
    dict(h=0.05,hs=0.020,s=0.75,ss=0.08,
         v=0.78,vs=0.08,sh=0.12,sr=0.32),

    # 500k
    dict(h=0.47,hs=0.030,s=0.65,ss=0.08,
         v=0.60,vs=0.08,sh=0.58,sr=0.15),
]

def build_training_data(n_per=400, px=300, seed=42):
    """
    Sinh features TRỰC TIẾP từ phân phối màu — không tạo ảnh trung gian.
    n_per=400 × 6 lớp = 2400 mẫu, px=300 pixel/mẫu.
    Chạy < 0.2 giây.
    """
    rng = np.random.default_rng(seed)
    Xs, ys = [], []

    for lbl, p in enumerate(_PROFILES):
        # lighting shift ngẫu nhiên cho từng mẫu
        h_shift = rng.normal(0, 0.012, (n_per, 1))
        s_shift = rng.normal(0, 0.07,  (n_per, 1))
        v_shift = rng.normal(0, 0.07,  (n_per, 1))

        # chọn primary vs secondary hue
        use_sec = rng.random((n_per, px)) < p['sr']
        h_c     = np.where(use_sec, p['sh'], p['h'] + h_shift)
        h_std   = np.where(use_sec, 0.03,   p['hs'])

        H = np.clip(rng.normal(h_c, h_std), 0, 1) % 1
        S = np.clip(rng.normal(p['s'] + s_shift, p['ss'], (n_per, px)), 0.04, 0.98)
        V = np.clip(rng.normal(p['v'] + v_shift, p['vs'], (n_per, px)), 0.10, 0.96)

        # glare & shadow
        glare  = rng.random((n_per, px)) < 0.08
        shadow = rng.random((n_per, px)) < 0.05
        S = np.where(glare,  S * 0.2,  S)
        V = np.where(glare,  np.clip(V * 1.3 + 0.15, 0, 0.98), V)
        V = np.where(shadow, V * rng.uniform(0.35, 0.55, (n_per, px)), V)

        valid = (V > 0.10) & (V < 0.96) & (S > 0.05)
        n     = valid.sum(axis=1, keepdims=True).clip(1, None).astype(np.float32)

        # histograms (vectorized)
        def histo(arr, bins, lo=0.0, hi=1.0):
           edges = np.linspace(lo, hi, bins + 1)
           out = np.zeros((n_per, bins), dtype=np.float32)

           for b in range(bins):

        # bin cuối lấy luôn giá trị = hi (vd 1.0)
               if b == bins - 1:
                  mask = (arr >= edges[b]) & (arr <= edges[b+1])
               else:
                  mask = (arr >= edges[b]) & (arr < edges[b+1])

               out[:, b] = (mask & valid).sum(1)

           return out / n

        h_hist = histo(H, H_BINS)
        s_hist = histo(S, S_BINS)
        v_hist = histo(V, V_BINS)

        # color ratios
        # color ratios
        def ratio(mask):
           return (mask & valid).sum(1, keepdims=True) / n

        red    = ratio((H < 0.05) | (H > 0.94))
        orange = ratio((H >= 0.04) & (H < 0.11))
        yellow = ratio((H >= 0.10) & (H < 0.17))
        green  = ratio((H >= 0.20) & (H < 0.42))
        teal   = ratio((H >= 0.42) & (H < 0.55))
        blue   = ratio((H >= 0.55) & (H < 0.72))

        mean_h = (H * valid).sum(1, keepdims=True) / n
        mean_s = (S * valid).sum(1, keepdims=True) / n
        mean_v = (V * valid).sum(1, keepdims=True) / n

        hi_sat = ratio(S > 0.5)

# synthetic edge/texture approximation
        edge_density = np.clip(
    0.25 + 0.35*green + 0.15*blue,
    0,1
)

        texture = np.clip(
        0.4 + 0.8*hi_sat,
        0,2
)
        # synthetic ORB approximation
        orb_density = np.clip(
    0.03 + 0.12*edge_density,
    0, 0.3
)

        mean_resp = np.clip(
    0.01 + 0.6*texture,
    0, 1
)
        dark   = np.clip(1.0 - mean_v, 0,1)
        mid    = np.clip(1.0 - np.abs(mean_v-0.5)*2, 0,1)
        bright = np.clip(mean_v,0,1)

        extra = np.hstack([
    red,orange,yellow,green,teal,blue,

    mean_h,mean_s,mean_v,

    edge_density,
    texture,

    orb_density,
    mean_resp,

    dark,
    mid,
    bright
]).astype(np.float32)
        feat  = np.concatenate([h_hist, s_hist, v_hist, extra], axis=1)
        Xs.append(feat)
        ys.extend([lbl] * n_per)

    X = np.vstack(Xs)
    y = np.array(ys, dtype=np.int32)
    mean = X.mean(0); std = X.std(0) + 1e-8
    return (X - mean) / std, y, mean, std

def _relu(x):   return np.maximum(0.0, x)
def _relu_d(z): return (z > 0).astype(np.float32)
def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

class ANN:
    def __init__(self, sizes, seed=42):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes)-1):
            sc = np.sqrt(2.0 / sizes[i])
            self.W.append(rng.normal(0, sc, (sizes[i], sizes[i+1])).astype(np.float32))
            self.b.append(np.zeros(sizes[i+1], dtype=np.float32))
        self.L = len(self.W)

    def _fwd(self, X):
        acts, zs = [X], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b; zs.append(z)
            acts.append(_softmax(z) if i == self.L-1 else _relu(z))
        return acts, zs

    def predict_proba(self, X):
        return self._fwd(np.atleast_2d(X))[0][-1]

    def predict(self, X):
        return int(np.argmax(self.predict_proba(X)))

    def fit(self, X, y, epochs=120, lr=0.025, batch=64):
        n = X.shape[0]
        rng = np.random.default_rng(0)
        for ep in range(epochs):
            cur_lr = lr * (0.96 ** (ep // 12))
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                bx = X[idx[s:s+batch]]; by = y[idx[s:s+batch]]
                acts, zs = self._fwd(bx); bn = bx.shape[0]
                yoh = np.zeros_like(acts[-1]); yoh[np.arange(bn), by] = 1.0
                delta = (acts[-1] - yoh) / bn
                for i in reversed(range(self.L)):

    # lưu W trước khi update
                    W_old = self.W[i].copy()

    # gradient descent
                    self.W[i] -= cur_lr * (acts[i].T @ delta)
                    self.b[i] -= cur_lr * delta.sum(0)

    # backprop cho layer trước
                    if i > 0:
                        delta = (delta @ W_old.T) * _relu_d(zs[i-1])

class Recognizer:
    def __init__(self):
        self.ready = False

        self.orb = cv2.ORB_create(nfeatures=300)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.templates = {}
    def load_templates(self):

      for d in DENOMS:

        path = f"templates/{d}.png"

        img = cv2.imread(path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp, des = self.orb.detectAndCompute(gray, None)

        self.templates[d] = des
    def orb_scores(self, img_bgr):

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        kp, des = self.orb.detectAndCompute(gray, None)

        scores = []

        if des is None:
            return [0]*len(DENOMS)

        for d in DENOMS:

            des_t = self.templates.get(d)

            if des_t is None:
               scores.append(0)
               continue

            matches = self.bf.match(des, des_t)

            matches = sorted(matches, key=lambda x: x.distance)

            good = [m for m in matches if m.distance < 55]

            scores.append(len(good))

        return scores
    def train(self):
        X = []
        y = []

        for lbl, d in enumerate(DENOMS):

          img = cv2.imread(f"templates/{d}.png")
          
          if img is None:
             continue

          feat = extract_features(img)

          orb = np.array(
           self.orb_scores(img),
        dtype=np.float32
    )

          feat = np.concatenate([feat, orb])

          for _ in range(120):
              X.append(feat)
              y.append(lbl)

        X = np.array(X,np.float32)
        y = np.array(y,np.int32)

        self.mean = X.mean(0)
        self.std = X.std(0)+1e-8

        X = (X-self.mean)/self.std
        self.ann = ANN([N_FEAT, 128, 64, 32, len(DENOMS)])
        self.ann.fit(
    X,
    y,
    epochs=180,
    lr=0.015,
    batch=64
)
        self.ready = True

    def predict(self, img_bgr):
        feat = extract_features(img_bgr)

        orb = np.array(
            self.orb_scores(img_bgr),
            dtype=np.float32
)

        feat = np.concatenate([feat, orb])

        feat = (feat - self.mean) / self.std
        p    = self.ann.predict_proba(feat.reshape(1,-1))[0]
        best = int(np.argmax(p))
        return {"denomination": DENOMS[best], "confidence": float(p[best]),
                "all_confidences": {DENOMS[i]: float(v) for i,v in enumerate(p)}}

def hex_rgba(h, a=0.10):
    r,g,b = int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return f"rgba({r},{g},{b},{a})"

st.set_page_config(page_title="Nhận dạng tiền VN", page_icon="💵", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Be Vietnam Pro',sans-serif!important}
.main{background:#f7f8fc}
[data-testid="stFileUploadDropzone"]{border:2px dashed #ccc!important;border-radius:14px!important;background:#fff!important}
.stButton>button{border-radius:50px!important;font-weight:700!important;border:2px solid #ddd!important}
div[data-testid="stExpander"]{border:1px solid #e8e8e8!important;border-radius:12px!important;background:#fff!important}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align:center;padding:28px 0 8px">
  <div style="font-size:2.8rem">🏦</div>
  <h1 style="font-size:1.85rem;font-weight:800;margin:4px 0;
     background:linear-gradient(135deg,#16a085,#1a6eb5);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent">
     Nhận dạng tiền giấy Việt Nam</h1>
  <p style="color:#888;font-size:0.88rem;margin:4px 0 0">
     ANN · 56 đặc trưng màu + texture · Huấn luyện &lt; 4 giây
</div>
""", unsafe_allow_html=True)
st.divider()

@st.cache_resource(show_spinner=False)
def load_model():
    r = Recognizer(); r.train(); return r

prog = st.progress(0, text="⚙️ Đang huấn luyện ANN...")
t0 = time.time()
prog.progress(30, text="⚙️ Sinh dữ liệu vectorized...")
model = load_model()
prog.progress(100, text="✅ Sẵn sàng!")
elapsed = time.time() - t0
time.sleep(0.4); prog.empty()
if elapsed > 0.5:
    st.success(f"✅ ANN huấn luyện xong trong **{elapsed:.1f}s**")

# Expander info
with st.expander("📖 Đặc điểm màu từng mệnh giá", expanded=False):
    cols = st.columns(3)
    for i, d in enumerate(DENOMS):
        c = COLORS[d]
        cols[i%3].markdown(f"""
        <div style="border-left:4px solid {c};background:{hex_rgba(c)};
                    padding:8px 12px;border-radius:8px;margin-bottom:8px">
          <b style="color:{c}">{NAMES[d]}</b><br>
          <small style="color:#555">🎨 {DESC[d][0]} · 🏛 {DESC[d][1]}</small>
        </div>""", unsafe_allow_html=True)

st.divider()

# Upload
col1, col2 = st.columns([3,2])
with col1:
    uploaded = st.file_uploader("📎 Tải ảnh tờ tiền", type=["jpg","jpeg","png","webp"])
with col2:
    use_cam = st.checkbox("📷 Dùng camera")
    cam_img = st.camera_input("Chụp tờ tiền") if use_cam else None

source = cam_img or uploaded

if source:
    pil = Image.open(source).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    left, right = st.columns([1,1], gap="medium")
    with left:
        st.image(pil, caption="Ảnh tải lên", use_container_width=True)
    with right:
        with st.spinner("🔍 Đang nhận dạng..."):
            res = model.predict(bgr)
        d, conf = res["denomination"], res["confidence"]
        c = COLORS[d]; bg = hex_rgba(c, 0.10)
        tone, loc = DESC[d]
        st.markdown(f"""
        <div style="background:{bg};border:2px solid {c};border-radius:16px;
                    padding:20px;text-align:center">
          <div style="font-size:2rem">{EMOJI[d]}</div>
          <div style="font-size:0.72rem;color:#888;letter-spacing:1.5px;
                      text-transform:uppercase;margin:4px 0">Kết quả</div>
          <div style="font-size:1.85rem;font-weight:800;color:{c}">{NAMES[d]}</div>
          <div style="font-size:0.85rem;color:#666;margin:4px 0 12px">
              <b>{tone}</b> · {loc}</div>
          <div style="background:{c};color:#fff;border-radius:50px;
                      padding:5px 20px;display:inline-block;
                      font-size:1.05rem;font-weight:700">
              {conf*100:.1f}% tin cậy</div>
        </div>""", unsafe_allow_html=True)
        if conf < 0.60:
            st.warning("⚠️ Độ tin cậy thấp — chụp rõ hơn, nền tối, đủ sáng.")

    st.divider()
    tab1, tab2 = st.tabs(["📊 Xác suất", "🔬 Chi tiết"])

    with tab1:
        st.subheader("Xác suất ANN theo từng mệnh giá")
        for dd, prob in sorted(res["all_confidences"].items(), key=lambda x:-x[1]):
            c2 = COLORS[dd]; pct = prob*100
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;
                        margin-bottom:7px;padding:7px 12px;
                        background:{hex_rgba(c2,0.07)};border-radius:10px">
              <span style="min-width:110px;font-size:0.82rem;
                           color:#333;font-weight:600">{NAMES[dd]}</span>
              <div style="flex:1;background:#e8e8e8;border-radius:5px;height:9px;overflow:hidden">
                <div style="width:{max(2,pct):.1f}%;background:{c2};height:100%;border-radius:5px"></div>
              </div>
              <span style="min-width:46px;text-align:right;font-size:0.82rem;
                           color:{c2};font-weight:700">{pct:.1f}%</span>
            </div>""", unsafe_allow_html=True)

        # matplotlib chart
        fig, ax = plt.subplots(figsize=(7,2.8), facecolor="none")
        labs = [NAMES[d2] for d2 in DENOMS]
        vals = [res["all_confidences"][d2]*100 for d2 in DENOMS]
        cols_plt = [COLORS[d2] for d2 in DENOMS]
        bars = ax.barh(labs, vals, color=cols_plt, height=0.55,
                       edgecolor="white", linewidth=0.8)
        ax.set_xlim(0,108); ax.spines[["top","right","left"]].set_visible(False)
        ax.xaxis.grid(True, ls="--", alpha=0.3, color="#ccc"); ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9, colors="#333")
        ax.tick_params(axis="x", labelsize=8, colors="#999")
        ax.set_xlabel("Xác suất (%)", fontsize=9, color="#666")
        for bar, v in zip(bars, vals):
            ax.text(v+1, bar.get_y()+bar.get_height()/2,
                    f"{v:.1f}%", va="center", fontsize=8, color="#444")
        fig.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with tab2:
        feat = extract_features(bgr)
        c1, c2 = st.columns(2)
        offset = H_BINS + S_BINS + V_BINS
        labels_c = ["Đỏ","Cam","Vàng","Xanh lá","Teal","Xanh lam"]
        with c1:
            st.markdown("**🌈 Tỉ lệ vùng màu**")
            for i, lbl in enumerate(labels_c):
                v = float(feat[offset+i])
                st.progress(min(int(v*100),100), text=f"{lbl}: {v*100:.1f}%")
        with c2:
            st.metric("Mean Hue",        f"{feat[offset+6]*360:.1f}°")
            st.metric("Mean Saturation", f"{feat[offset+7]*100:.1f}%")
            st.metric("Mean Value",      f"{feat[offset+8]*100:.1f}%")
            st.metric("Texture",         f"{feat[offset+10]:.2f}")

        st.divider()
        st.table({"Mệnh giá":[NAMES[d2] for d2 in DENOMS],
                  "Xác suất":[f"{res['all_confidences'][d2]*100:.2f}%" for d2 in DENOMS]})

else:
    st.markdown("""
    <div style="text-align:center;padding:16px 0;color:#aaa;font-size:0.95rem">
        📎 Tải ảnh tờ tiền lên để bắt đầu
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 💰 Các mệnh giá được hỗ trợ")
    cols = st.columns(6)
    for i, d in enumerate(DENOMS):
        c = COLORS[d]
        cols[i].markdown(f"""
        <div style="text-align:center;background:{hex_rgba(c,0.10)};
                    border:2px solid {c};border-radius:12px;padding:12px 4px">
          <div style="font-size:1.3rem">{EMOJI[d]}</div>
          <b style="color:{c};font-size:0.78rem">{NAMES[d]}</b>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<br>
    <div style="background:#fff7e6;border:1px solid #ffe0a0;border-radius:12px;
                padding:14px 18px;font-size:0.88rem;color:#8a6200">
      💡 <b>Mẹo:</b> Đặt tờ tiền trên nền tối, đủ ánh sáng, tránh bóng & lóa.
    </div>""", unsafe_allow_html=True)

st.markdown("""<hr style="margin-top:40px;border-color:#eee">
<p style="text-align:center;color:#ccc;font-size:0.75rem">
ANN thuần NumPy · HSV Feature Extraction · Streamlit UI</p>""",
unsafe_allow_html=True)

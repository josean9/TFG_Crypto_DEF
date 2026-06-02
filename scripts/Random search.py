"""
Búsqueda aleatoria de subconjuntos de features para LSTM (retorno ETH a 3 días).
Diseñado para dejar corriendo en GPU toda la noche. Guarda cada modelo en CSV
inmediatamente -> si se corta, no se pierde lo hecho.

USO en el ordenador de la universidad:
    python random_search.py
Ajusta arriba RUTA_DF, RUTA_REGIMENES, N_MODELOS y SALIDA_CSV.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, time, os, random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, StandardScaler

# ─────────── CONFIG (ajustar en la universidad) ───────────
RUTA_DF        = "../data/csv/df_merged.csv"     # tu dataset (tras la ingeniería de features ya hecha)
RUTA_REGIMENES = "../data/csv/regimenes.csv"
SALIDA_CSV     = "random_search_resultados.csv"
N_MODELOS      = 4000
MIN_FEATS, MAX_FEATS = 3, 25      # rango de tamaño de subconjunto
SEQ_LEN, HORIZON = 30, 3
SEMILLA_GLOBAL = 42
# ───────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")

# Para el smoke test local generamos datos sintéticos si no existe el CSV
if not os.path.exists(RUTA_DF):
    print("CSV no encontrado -> generando datos sintéticos para PROBAR el script")
    np.random.seed(0); N=600
    cols = [f"feat{i}" for i in range(40)]
    df_model = pd.DataFrame(np.random.randn(N,40), columns=cols)
    df_model["eth_close_ret"] = np.random.randn(N)*2.5
    TARGET_COL="eth_close_ret"
    FEATURE_COLS = cols + [TARGET_COL]
else:
    df_model = pd.read_csv(RUTA_DF, parse_dates=["date"], index_col="date")
    TARGET_COL = "eth_close_ret"
    # aquí en producción irían tus EXCLUIR y la integración de régimen;
    # para el script asumimos que df_model ya tiene las features y el régimen
    FEATURE_COLS = [c for c in df_model.columns if c == TARGET_COL or not c.startswith(("btc_open","btc_high"))]
    FEATURE_COLS = [c for c in df_model.columns]  # simplificado; ajustar a tu EXCLUIR real

# split temporal
n=len(df_model); i_tr=int(n*0.70); i_va=int(n*0.85)
df_tr, df_va = df_model.iloc[:i_tr], df_model.iloc[i_tr:i_va]

class LSTMReg(nn.Module):
    def __init__(self, nf, h=(48,24), horizon=HORIZON, dropout=0.35):
        super().__init__()
        capas=[]; ins=nf
        for hh in h: capas.append(nn.LSTM(ins,hh,batch_first=True)); ins=hh
        self.lstms=nn.ModuleList(capas)
        self.drops=nn.ModuleList([nn.Dropout(dropout) for _ in h])
        self.head=nn.Sequential(nn.Linear(h[-1],max(h[-1]//2,1)),nn.ReLU(),
                                nn.Dropout(dropout),nn.Linear(max(h[-1]//2,1),horizon))
    def forward(self,x):
        out=x
        for l,d in zip(self.lstms,self.drops): out,_=l(out); out=d(out)
        return self.head(out[:,-1,:])

class DS(Dataset):
    def __init__(s,X,y): s.X=torch.FloatTensor(X); s.y=torch.FloatTensor(y)
    def __len__(s): return len(s.X)
    def __getitem__(s,i): return s.X[i],s.y[i]

def secuencias(X,y):
    Xs,ys=[],[]
    for i in range(SEQ_LEN,len(X)-HORIZON+1):
        Xs.append(X[i-SEQ_LEN:i]); ys.append(y[i:i+HORIZON])
    return np.array(Xs,np.float32),np.array(ys,np.float32)

def evaluar_subconjunto(cols_sel, sy):
    # escalado solo con train
    sx=RobustScaler().fit(df_tr[cols_sel].values)
    Xtr=sx.transform(df_tr[cols_sel].values); Xva=sx.transform(df_va[cols_sel].values)
    ytr=sy.transform(df_tr[TARGET_COL].values.reshape(-1,1)).ravel()
    yva=sy.transform(df_va[TARGET_COL].values.reshape(-1,1)).ravel()
    Xs_tr,ys_tr=secuencias(Xtr,ytr); Xs_va,ys_va=secuencias(Xva,yva)
    tl=DataLoader(DS(Xs_tr,ys_tr),batch_size=32,shuffle=True)
    vl=DataLoader(DS(Xs_va,ys_va),batch_size=64,shuffle=False)
    m=LSTMReg(len(cols_sel)).to(DEVICE)
    crit=nn.HuberLoss(); opt=torch.optim.Adam(m.parameters(),lr=8e-4,weight_decay=3e-4)
    mejor=float("inf"); sin=0
    for ep in range(40):
        m.train()
        for xb,yb in tl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
            loss=crit(m(xb),yb); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval(); v=0
        with torch.no_grad():
            for xb,yb in vl:
                xb,yb=xb.to(DEVICE),yb.to(DEVICE); v+=crit(m(xb),yb).item()*len(xb)
        v/=len(vl.dataset)
        if v<mejor-1e-5: mejor=v; sin=0
        else:
            sin+=1
            if sin>=6: break
    # DirAcc en val
    m.eval(); preds=[];trues=[]
    with torch.no_grad():
        for xb,yb in vl:
            preds.append(m(xb.to(DEVICE)).cpu().numpy()); trues.append(yb.numpy())
    preds=np.concatenate(preds).ravel(); trues=np.concatenate(trues).ravel()
    da=np.mean(np.sign(trues)==np.sign(preds))
    return mejor, da

# ── bucle principal ──
random.seed(SEMILLA_GLOBAL); np.random.seed(SEMILLA_GLOBAL); torch.manual_seed(SEMILLA_GLOBAL)
feats_disponibles=[c for c in FEATURE_COLS if c!=TARGET_COL]
sy=StandardScaler().fit(df_tr[TARGET_COL].values.reshape(-1,1))

# cabecera CSV (si no existe)
if not os.path.exists(SALIDA_CSV):
    with open(SALIDA_CSV,"w") as f: f.write("modelo,n_feats,val_loss,dir_acc,features\n")

t0=time.time()
N_PRUEBA = N_MODELOS
for k in range(N_PRUEBA):
    if k % 50 == 0:
        print(f"  [{k}/{N_MODELOS}] {(time.time()-t0)/60:.1f} min transcurridos", flush=True)
    nf=random.randint(MIN_FEATS, min(MAX_FEATS,len(feats_disponibles)))
    cols_sel=random.sample(feats_disponibles, nf)
    try:
        vl_loss, da = evaluar_subconjunto(cols_sel, sy)
        with open(SALIDA_CSV,"a") as f:
            f.write(f"{k},{nf},{vl_loss:.5f},{da:.4f},\"{'|'.join(cols_sel)}\"\n")
    except Exception as e:
        print(f"  modelo {k} falló: {e}")
print(f"Smoke test: {N_PRUEBA} modelos en {time.time()-t0:.1f}s -> {(time.time()-t0)/N_PRUEBA:.1f}s/modelo")

# analizar lo guardado
r=pd.read_csv(SALIDA_CSV)
print(f"\nCSV tiene {len(r)} filas. Mejor por val_loss:")
print(r.sort_values("val_loss").head(3)[["n_feats","val_loss","dir_acc"]].to_string(index=False))
print("\nOK script funciona end-to-end")
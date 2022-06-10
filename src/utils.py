import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import numpy as np
import torch



def torch_fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


class DataSet:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index, :], self.y[index], self.z[index, :]

def ndarray_to_tensor(x):
        x = torch.tensor(x).float()
        return x

def fetch_sample_data(random_state=0, test_size=0.15, StandardScaler=False):
    RCT_DATA = "http://www.nber.org/~rdehejia/data/nsw_dw.dta"
    CPS_DATA = "http://www.nber.org/~rdehejia/data/cps_controls3.dta"  # cps調査で直前に無職だったものに限定

    df = pd.concat(
        [
            pd.read_stata(RCT_DATA).query("treat>0"),  # (失業者介入実験データ)nsw data の介入群のみ抽出
            pd.read_stata(CPS_DATA),  # 別のセンサスデータ
        ]
    ).reset_index(drop=True)

    del df["data_id"]

    df["treat"] = df["treat"].astype(int)
    if StandardScaler:
        features_cols = [col for col in df.columns if col not in ["treat", "re78"]]
        ss = preprocessing.StandardScaler()
        df_std = pd.DataFrame(ss.fit_transform(df[features_cols]), columns=features_cols)
        df_std = pd.concat([df[["treat", "re78"]], df_std], axis=1)
        df = df_std.copy()
    
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        df.drop(["re78", "treat"], axis=1),
        df[["re78"]],
        df[["treat"]],
        random_state=random_state,
        test_size=test_size,
    )


    X_train = torch.FloatTensor(X_train.to_numpy())
    y_train = torch.FloatTensor(y_train.to_numpy())
    t_train = torch.FloatTensor(t_train.to_numpy())

    X_test = torch.FloatTensor(X_test.to_numpy())
    y_test = torch.FloatTensor(y_test.to_numpy())
    t_test = torch.FloatTensor(t_test.to_numpy())

    dataset = DataSet(X_train, y_train, t_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True, drop_last=True)

    return  dataloader, X_train, y_train, t_train, X_test, y_test, t_test

def mmd_rbf(Xt, Xc, p, sig=0.1):
    sig = torch.tensor(sig)
    Kcc = torch.exp(-torch.cdist(Xc, Xc, 2.0001) / torch.sqrt(sig))
    Kct = torch.exp(-torch.cdist(Xc, Xt, 2.0001) / torch.sqrt(sig))
    Ktt = torch.exp(-torch.cdist(Xt, Xt, 2.0001) / torch.sqrt(sig))

    m = Xc.shape[0]
    n = Xt.shape[0]

    mmd = (1 - p) ** 2 / (m *(m-1)) * (Kcc.sum() - m)
    mmd += p ** 2 / (n * (n-1)) * (Ktt.sum() - n)
    mmd -= 2 * p * (1 - p) / (m * n) * Kct.sum()
    mmd *= 4

    return mmd

def mmd_lin(Xt, Xc, p):
    mean_treated = torch.mean(Xt)
    mean_control = torch.mean(Xc)
    
    mmd = torch.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control).sum()

    return mmd


def ipm_scores(model, X, _t, sig=0.1):
    _t_id = np.where((_t.cpu().detach().numpy() == 1).all(axis=1))[0]
    _c_id = np.where((_t.cpu().detach().numpy() == 0).all(axis=1))[0]
    x_rep = model.repnet(X)
    ipm_lin = mmd_lin(
                    x_rep[_t_id],
                    x_rep[_c_id],
                    p=len(_t_id) / (len(_t_id) + len(_c_id))
    )
    ipm_rbf = mmd_rbf(
                    x_rep[_t_id],
                    x_rep[_c_id],
                    p=len(_t_id) / (len(_t_id) + len(_c_id)),
                    sig=sig,
                )
    ipm_lin_pre = mmd_lin(
                    X[_t_id],
                    X[_c_id],
                    p=len(_t_id) / (len(_t_id) + len(_c_id))
    )
    ipm_rbf_pre = mmd_rbf(
                    X[_t_id],
                    X[_c_id],
                    p=len(_t_id) / (len(_t_id) + len(_c_id)),
                    sig=sig,
                )
    return {
        "ipm_lin": np.float64(ipm_lin.cpu().detach().numpy()),
        "ipm_rbf": np.float64(ipm_rbf.cpu().detach().numpy()),
        "ipm_lin_before": np.float64(ipm_lin_pre.cpu().detach().numpy()),
        "ipm_rbf_before": np.float64(ipm_rbf_pre.cpu().detach().numpy()),
    }
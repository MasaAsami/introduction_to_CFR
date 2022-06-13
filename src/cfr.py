import sys
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from src.mlp import MLP
from src.utils import mmd_rbf, mmd_lin


def get_score(model, x_test, y_test, t_test):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    N = len(x_test)

    # MSE
    _ypred = model.forward(x_test, t_test)
    mse = mean_squared_error(y_test, _ypred)

    # treatment index
    t_idx = np.where(t_test.to("cpu").detach().numpy().copy() == 1)[0]
    c_idx = np.where(t_test.to("cpu").detach().numpy().copy() == 0)[0]

    # ATE & ATT
    _t0 = torch.FloatTensor([0 for _ in range(N)]).reshape([-1, 1])
    _t1 = torch.FloatTensor([1 for _ in range(N)]).reshape([-1, 1])

    # _cate = model.forward(x_test, _t1) - model.forward(x_test, _t0)
    _cate_t = y_test - model.forward(x_test, _t0)
    _cate_c = model.forward(x_test, _t1) - y_test
    _cate = torch.cat([_cate_c[c_idx], _cate_t[t_idx]])
    
    _ate = np.mean(_cate.to("cpu").detach().numpy().copy())
    _att = np.mean(_cate_t[t_idx].to("cpu").detach().numpy().copy())

    return {"ATE": _ate, "ATT": _att, "RMSE": np.sqrt(mse)}


class Base(nn.Module):
    def __init__(self, cfg):
        super(Base, self).__init__()
        self.cfg = cfg
        self.criterion = nn.MSELoss(reduction='none')
        self.mse = mean_squared_error

    def fit(
        self,
        dataloader,
        x_train,
        y_train,
        t_train,
        x_test,
        y_test,
        t_test,
        logger,
    ):
        losses = []
        ipm_result = []
        logger.debug("                          within sample,      out of sample")
        logger.debug("           [Train MSE, IPM], [RMSE, ATT, ATE], [RMSE, ATT, ATE]")
        for epoch in range(self.cfg["epochs"]):
            epoch_loss = 0
            epoch_ipm = []
            n = 0
            for (x, y, z) in dataloader:

                x = x.to(device=torch.device("cpu"))
                y = y.to(device=torch.device("cpu"))
                z = z.to(device=torch.device("cpu"))
                self.optimizer.zero_grad()

                x_rep = self.repnet(x)

                _t_id = np.where((z.cpu().detach().numpy() == 1).all(axis=1))[0]
                _c_id = np.where((z.cpu().detach().numpy() == 0).all(axis=1))[0]

                if self.cfg["split_outnet"]:
                    y_hat_treated = self.outnet_treated(x_rep[_t_id])
                    y_hat_control = self.outnet_control(x_rep[_c_id])

                    _index = np.argsort(np.concatenate([_t_id, _c_id], 0))

                    y_hat = torch.cat([y_hat_treated, y_hat_control])[_index]
                else:
                    y_hat = self.outnet(torch.cat((x_rep, z), 1))

                loss = self.criterion(y_hat, y.reshape([-1, 1]))
                # sample weight
                p_t = np.mean(z.cpu().detach().numpy())
                w_t = z/(2*p_t)
                w_c = (1-z)/(2*1-p_t)
                sample_weight = w_t + w_c
                if (p_t ==1) or (p_t ==0):
                    sample_weight = 1
                
                loss =torch.mean((loss * sample_weight))

                if self.cfg["alpha"] > 0.0:    
                    if self.cfg["ipm_type"] == "mmd_rbf":
                       ipm = mmd_rbf(
                            x_rep[_t_id],
                            x_rep[_c_id],
                            p=len(_t_id) / (len(_t_id) + len(_c_id)),
                            sig=self.cfg["sig"],
                        )
                    elif self.cfg["ipm_type"] == "mmd_lin":
                        ipm = mmd_lin(
                            x_rep[_t_id],
                            x_rep[_c_id],
                            p=len(_t_id) / (len(_t_id) + len(_c_id))
                        )
                    else:
                        logger.debug(f'{self.cfg["ipm_type"]} : TODO!!! Not implemented yet!')
                        sys.exit()

                    loss += ipm * self.cfg["alpha"]
                    epoch_ipm.append(ipm.cpu().detach().numpy())

                
                mse = self.mse(
                    y_hat.detach().cpu().numpy(),
                    y.reshape([-1, 1]).detach().cpu().numpy(),
                )
                
                loss.backward()

                self.optimizer.step()
                epoch_loss += mse * y.shape[0]
                n += y.shape[0]

            self.scheduler.step()
            epoch_loss = epoch_loss / n
            losses.append(epoch_loss)
            if self.cfg["alpha"] > 0:
                ipm_result.append(np.mean(epoch_ipm))

            if epoch % 100 == 0:
                with torch.no_grad():
                    within_result = get_score(self, x_train, y_train, t_train)
                    outof_result = get_score(self, x_test, y_test, t_test)
                logger.debug(
                    "[Epoch: %d] [%.3f, %.3f], [%.3f, %.3f, %.3f], [%.3f, %.3f, %.3f] "
                    % (
                        epoch,
                        epoch_loss,
                        ipm if self.cfg["alpha"] > 0 else -1,
                        within_result["RMSE"],
                        within_result["ATT"],
                        within_result["ATE"],
                        outof_result["RMSE"],
                        outof_result["ATT"],
                        outof_result["ATE"],
                    )
                )

        return within_result, outof_result, losses, ipm_result


class CFR(Base):
    def __init__(self, in_dim, out_dim, cfg={}):
        super().__init__(cfg)

        self.repnet = MLP(
            num_layers=cfg["repnet_num_layers"],
            in_dim=in_dim,
            hidden_dim=cfg["repnet_hidden_dim"],
            out_dim=cfg["repnet_out_dim"],
            activation=nn.ReLU(inplace=True),
            dropout=cfg["repnet_dropout"],
        )

        if cfg["split_outnet"]:

            self.outnet_treated = MLP(
                in_dim=cfg["repnet_out_dim"], out_dim=out_dim, num_layers=cfg["outnet_num_layers"], hidden_dim=cfg["outnet_hidden_dim"], dropout=cfg["outnet_dropout"]
            )
            self.outnet_control = MLP(
                in_dim=cfg["repnet_out_dim"], out_dim=out_dim, num_layers=cfg["outnet_num_layers"], hidden_dim=cfg["outnet_hidden_dim"], dropout=cfg["outnet_dropout"]
            )

            self.params = (
                list(self.repnet.parameters())
                + list(self.outnet_treated.parameters())
                + list(self.outnet_control.parameters())
            )
        else:
            self.outnet = MLP(
                in_dim=cfg["repnet_out_dim"] + 1, out_dim=out_dim, num_layers=cfg["outnet_num_layers"], hidden_dim=cfg["outnet_hidden_dim"], dropout=cfg["outnet_dropout"]
            )

            self.params = (
                list(self.repnet.parameters())
                + list(self.outnet.parameters())
            )

        self.optimizer = optim.Adam(
            params=self.params, lr=cfg["lr"], weight_decay=cfg["wd"]
        )
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=cfg["gamma"])

    def forward(self, x, z):
        with torch.no_grad():
            x_rep = self.repnet(x)

            if self.cfg["split_outnet"]:

                _t_id = np.where((z.cpu().detach().numpy() == 1).all(axis=1))[0]
                _c_id = np.where((z.cpu().detach().numpy() == 0).all(axis=1))[0]

                y_hat_treated = self.outnet_treated(x_rep[_t_id])
                y_hat_control = self.outnet_control(x_rep[_c_id])

                _index = np.argsort(np.concatenate([_t_id, _c_id], 0))
                y_hat = torch.cat([y_hat_treated, y_hat_control])[_index]
            else:
                y_hat = self.outnet(torch.cat((x_rep, z), 1))

        return y_hat

if __name__ == "__main__":
    from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG
    from datetime import datetime

    from utils import torch_fix_seed, fetch_sample_data, cal_ipm

    torch_fix_seed()
    
    logger = getLogger("run DFR")
    logger.setLevel(DEBUG)
    handler_format = Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(handler_format)
    file_handler = FileHandler(
        "logs/"
        + "cfr"
        + "-"
        + "{:%Y-%m-%d-%H:%M:%S}.log".format(datetime.now()),
        "a",
    )
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.debug("Start process.")

    dataloader, X_train, y_train, t_train, X_test, y_test, t_test = fetch_sample_data(random_state=0, test_size=0.15, StandardScaler=False)

    cfgs = {
        "lr": 1e-3, "wd": 0.5, "alpha": 10**8, "sig": 0.1,
        "epochs": 1000, "ipm_type": "mmd_rbf",
        "repnet_num_layers":3, "repnet_hidden_dim":48,
        "repnet_out_dim":48, "repnet_dropout":0.145,
        "outnet_num_layers":3, "outnet_hidden_dim":32,
        "outnet_dropout":0.145, "gamma":0.97, "split_outnet": True
        }
    model = CFR(in_dim=8, out_dim=1, cfg=cfgs)
    within_result, outof_result, train_mse, ipm_result = model.fit(
        dataloader, X_train, y_train, t_train, X_test, y_test, t_test, logger
    )
    print(within_result)
    print(outof_result)

    _t_id = np.where((t_train.cpu().detach().numpy() == 1).all(axis=1))[0]
    _c_id = np.where((t_train.cpu().detach().numpy() == 0).all(axis=1))[0]
    x_rep = model.repnet(X_train)
    ipm_lin = mmd_lin(
                    x_rep[_t_id],
                    x_rep[_c_id],
                    p=len(_t_id) / (len(_t_id) + len(_c_id))
    )
    ipm_rbf = mmd_rbf(
                    x_rep[_t_id],
                    x_rep[_c_id],
                    p=len(_t_id) / (len(_t_id) + len(_c_id)),
                    sig=0.1,
                )
    ipm_lin = mmd_lin(
                    X_train[_t_id],
                    X_train[_c_id],
                    p=len(_t_id) / (len(_t_id) + len(_c_id))
    )
    ipm_rbf = mmd_rbf(
                    X_train[_t_id],
                    X_train[_c_id],
                    p=len(_t_id) / (len(_t_id) + len(_c_id)),
                    sig=0.1,
                )
    cal_ipm(model, X_train, t_train, sig=0.1)
    cal_ipm(model, X_test, t_test, sig=0.1)






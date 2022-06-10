import yaml
import itertools

alpha_list = [0, 0.5, 1, 100, 10000]
ipm_type_list = ["mmd_lin", "mmd_rbf"]
split_outnet = [True, False]

cfg_experiments = itertools.product(alpha_list, ipm_type_list, split_outnet)

for i, _cfg in enumerate(cfg_experiments):
    cfg_temp = {
            "alpha": _cfg[0],
            "lr": 1e-3,
            "wd": 0.5,
            "sig": 0.1,
            "epochs": 1000,
            "ipm_type": _cfg[1],
            "repnet_num_layers": 3,
            "repnet_hidden_dim": 48,
            "repnet_out_dim": 48,
            "repnet_dropout": 0.145,
            "outnet_num_layers": 3,
            "outnet_hidden_dim": 32,
            "outnet_dropout": 0.145,
            "gamma": 0.97,
            "split_outnet": _cfg[2],
            "experiments": f"experiments{i}",
            "StandardScaler": True
        }


    with open(f"configs/experiments{i}.yaml", "w") as yf:
        yaml.dump(
            cfg_temp,
            yf,
            default_flow_style=False,
        )

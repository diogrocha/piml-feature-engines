# Data

No data are included. The notebook and `get_data.sh` download them from public mirrors of
the official files; to use the official sources instead, place the files as below, keeping the original
names.

| Dataset | Folder | Files | Official source |
|---|---|---|---|
| C-MAPSS | `data/CMAPSSData/` | `train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt` | NASA Prognostics Center of Excellence data repository |
| Azure predictive maintenance | `data/azure/` | `PdM_telemetry.csv`, `PdM_maint.csv`, `PdM_failures.csv`, `PdM_errors.csv`, `PdM_machines.csv` | Microsoft Azure AI Gallery |
| NASA batteries | `data/nasa_battery/` | `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` | NASA Prognostics Center of Excellence data repository |

The C-MAPSS location can be changed with `LG_DATA=/path/to/CMAPSSData`; the Azure location with
`AZURE_DATA=/path/to/azure`. The controlled simulation needs no data.

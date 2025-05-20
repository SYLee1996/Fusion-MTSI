# Fusion-MTSI

Implementation of the paper **“Fusion‑MTSI: Fusion‑Based Multivariate Time Series Imputation”** (Journal of Advances in Information Technology, Vol. 16, No. 5, 2025).

PDF: [https://www.jait.us/articles/2025/JAIT-V16N5-666.pdf](https://www.jait.us/articles/2025/JAIT-V16N5-666.pdf)

---

## Project structure

```text
Fusion‑MTSI/
├── dataset/                 # Place all raw datasets here
│   ├── electricity.csv
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   ├── ETTm2.csv
│   ├── exchange_rate.csv
│   ├── national_illness.csv
│   ├── traffic.csv
│   └── weather.csv
├── Fusion_MTSI_MAIN.py      
├── Fusion_MTSI_MODEL.py     
├── Fusion_MTSI_UTILS.py     
├── requirements.txt         
└── README.md                
```

---

## Quick start

```bash
# 1. Clone the repo (or push this folder to GitHub first)
git clone https://github.com/<your‑account>/fusion‑mtsi.git
cd fusion‑mtsi

# 2. Create Python 3.12 environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run Fusion‑MTSI
python Fusion_MTSI_MAIN.py \
    --dataset 'weather' \
    --missing_rate 0.05 \
    --consecutive_missing_rate 0.05 \
    --max_missing_rate_per_feature 0.5 \
    --noise_rate 0.1 \
    \
    \
    \
    --model_name <set-filename-to-save> \
    --num_similar_features 3 \
    --n_neighbors 3 \
    --metric 'fusion_mtsi' \
    \
    \
    \
    --visualize
```

The main script saves imputed results and evaluation metrics to `RESULTS/`.

---

## Requirements

* Python **3.12**
* All Python packages pinned in `requirements.txt` – key libraries are:

  * numpy==1.26.4
  * pandas==2.2.1
  * scikit-learn==1.4.1.post1
  * numba==0.61.2
  * tqdm==4.66.1
  * matplotlib==3.10.3

To install a locked set of versions for full reproducibility you can also use:

```bash
pip install -r requirements.txt --no-cache-dir
```

---


## Dataset preparation

* Download raw CSV files for each dataset listed in the paper:

  * [Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
  * [Exchange](https://github.com/laiguokun/multivariate-time-series-data)
  * [Traffic](http://pems.dot.ca.gov/)
  * [Weather](https://www.bgc-jena.mpg.de/wetter/)
  * [ILI](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html)
  * [ETT (ETTh1, ETTm1, etc.)](https://github.com/zhouhaoyi/ETDataset)

---

## Citation

If you use this repository, please cite the original paper:

```bibtex
@article{lee2025fusionmtsi,
  title   = {Fusion-MTSI: Fusion-Based Multivariate Time Series Imputation},
  author  = {Lee, Sangyong and Hwang, Subo},
  journal = {Journal of Advances in Information Technology},
  volume  = {16},
  number  = {5},
  pages   = {666--675},
  year    = {2025},
  doi     = {10.12720/jait.16.5.666-675}
}
```

---

## License

Copyright © 2025 by the authors. This is an open access article distributed under the Creative Commons Attribution License which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited [(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

---

## Contact

* **Sangyong Lee** – [sangyong1996@gmail.com](mailto:sangyong1996@gmail.com)
* Issues and pull requests are welcome!

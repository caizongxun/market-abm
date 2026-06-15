# market-abm

Agent-Based Market Simulation — emergent K-bar generation from heterogeneous agent interactions.

## 一句話總結

讓 4 種交易 agent（機構、動能散戶、隨機散戶、逆勢者）在歷史市場條件下互相交易，
觀察撮合後的 K 棒統計特性是否與真實市場相似。

## 快速開始

```bash
pip install -r requirements.txt
python run_sim.py --symbol AAPL --bars 200 --plot
```

## 架構

```
market-abm/
├── sim/
│   ├── agents.py      # 4 種 agent：Institution / Momentum / Random / Contrarian
│   ├── market.py      # 市場撮合引擎，每輪產生一根 OHLCV K 棒
│   ├── simulation.py  # 主迴圈，跑 N 根
│   └── metrics.py     # 統計對比（分佈 / Hurst / 波動率叢聚）
├── data/
│   └── fetch.py       # yfinance 下載並快取
├── notebooks/
│   └── explore.ipynb  # 互動探索
├── run_sim.py         # 唯一入口
└── requirements.txt
```

## Agent 說明

| Agent | 邏輯 | 預設人數 | 資金規模 |
|---|---|---|---|
| `InstitutionAgent` | 均值回歸：收盤偏離 MA 越遠下單越大 | 5 | 大（×10）|
| `MomentumTrader` | 追漲殺跌：連續 N 根同向就跟進 | 40 | 中 |
| `RandomTrader` | 完全隨機買賣 | 100 | 小 |
| `ContrarianTrader` | 逆勢：漲多就空，跌多就多 | 15 | 中 |

## 驗證指標

- 日報酬分佈（偏度 / 峰度 / 尾部）
- Hurst 指數（趨勢慣性程度）
- 波動率自相關（GARCH 效果代理）
- 方向命中率

## 下一步

- [ ] 加入訂單簿撮合（取代市場衝擊模型）
- [ ] Agent 加入 VIX / 情緒資訊
- [ ] 參數掃描：不同 agent 比例下的市場特性變化
- [ ] 強化學習 agent（自適應策略）

## Changelog

---

### 0.1.0 Alpha (13.07.2022)

_Init versioning. At now project is in early WIP. Architecture of core logic is unstable and actively improving during further research. Use this project as a collection of drafts for good qunatitative trading system rather than ready-to-use modules._

List of features:

* Scalable event-based backtesting (__simple__ for single-asset tests and __advanced__ for multi-asset tests). Simple backtester use online data from Binance and run strategy in sync. Multi-asset backtester is written on multiprocessing and using ZeroMQ for inter-communication; data loaded from DB (need to sync first via command `poe sync`). 

* Basic strategy statistics and charts through exchange simulation.

* Simple and expandable interface for defining indicators and strategies.

* Storage layer: assets and candles data syncronization using Binance (considering migration to ByBit) and PostgreSQL (considering migration to TimescaleDB).

* Usage examples in Jupyter Notebooks.

* Some drafts for Web UI and real-time market data feeding.

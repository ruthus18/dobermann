## Changelog


### 0.3.0 Alpha (18.12.2022)

* Strategy lab module:
    * Feeding indicators with data
    * Simple backtesting
    * Account report calculation

* Visualization
    * Candlestick charts (simple and compound with indicators data/trades)
    * Account report charts (equity, used leverage and drawdowns)

* More indicators
    * MA variations (DEMA, TEMA)
    * Example of combining indicators (double MA crossing)
    * Example of working with multiple timescales (`TimeScale` indicator)

### 0.2.0 Alpha (06.11.2022)

Move to new project structure. Decide to clean up project and separate core functionality from experiments. Now old version lives in `drafts_` folder. New version lives in `app` folder.

Other updates:

* Switch to ByBit instead of Binance.

* Drafts of `Strategy Lab` module.

* Add example of basic working with candles data and indicators.


### 0.1.0 Alpha (13.07.2022)

_Init versioning. At now project is in early WIP. Architecture of core logic is unstable and actively improving during further research. Use this project as a collection of drafts for good qunatitative trading system rather than ready-to-use modules._

__List of features:__

* Scalable event-based backtesting (__simple__ for single-asset tests and __advanced__ for multi-asset tests). Simple backtester use online data from Binance and run strategy in sync. Multi-asset backtester is written on multiprocessing and using ZeroMQ for inter-communication; data loaded from DB (need to sync first via command `poe sync`). 

* Basic strategy statistics and charts through exchange simulation.

* Simple and expandable interface for defining indicators and strategies.

* Storage layer: assets and candles data syncronization using Binance (considering migration to ByBit) and PostgreSQL (considering migration to TimescaleDB).

* Usage examples in Jupyter Notebooks.

* Some drafts for Web UI and real-time market data feeding.

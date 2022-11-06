CREATE TABLE candles (
    id SERIAL PRIMARY KEY,
    asset VARCHAR(16) NOT NULL,
    timeframe VARCHAR(4) NOT NULL,
    open_time TIMESTAMP NOT NULL,
    open DECIMAL NOT NULL,
    close DECIMAL NOT NULL,
    low DECIMAL NOT NULL,
    high DECIMAL NOT NULL,
    volume DECIMAL NOT NULL,
    UNIQUE (asset, timeframe, open_time)
)

-- upgrade --
CREATE TABLE IF NOT EXISTS "candle" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "timeframe" VARCHAR(3) NOT NULL,
    "open_time" TIMESTAMPTZ NOT NULL,
    "open" DECIMAL(16,8) NOT NULL,
    "close" DECIMAL(16,8) NOT NULL,
    "low" DECIMAL(16,8) NOT NULL,
    "high" DECIMAL(16,8) NOT NULL,
    "volume" DECIMAL(24,8) NOT NULL,
    "asset_id" INT NOT NULL REFERENCES "asset" ("id") ON DELETE CASCADE,
    CONSTRAINT "uid_candle_asset_i_c5ab0e" UNIQUE ("asset_id", "timeframe", "open_time")
);
CREATE INDEX IF NOT EXISTS "idx_candle_timefra_6c3e0c" ON "candle" ("timeframe");
CREATE INDEX IF NOT EXISTS "idx_candle_asset_i_31dbeb" ON "candle" ("asset_id");
COMMENT ON COLUMN "candle"."timeframe" IS 'M1: 1m\nM3: 3m\nM5: 5m\nM15: 15m\nM30: 30m\nH1: 1h\nH2: 2h\nH4: 4h\nH6: 6h\nH8: 8h\nH12: 12h\nD1: 1d\nD3: 3d\nW1: 1w';
-- downgrade --
DROP TABLE IF EXISTS "candle";

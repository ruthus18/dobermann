-- upgrade --
CREATE TABLE IF NOT EXISTS "asset" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "ticker" VARCHAR(16) NOT NULL,
    "base_asset" VARCHAR(16) NOT NULL,
    "quote_asset" VARCHAR(16) NOT NULL,
    "status" VARCHAR(15) NOT NULL,
    "onboard_date" DATE NOT NULL,
    "filters" JSONB NOT NULL,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ,
    "removed_at" TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS "idx_asset_ticker_87aa4a" ON "asset" ("ticker");
COMMENT ON COLUMN "asset"."status" IS 'TRADING: TRADING\nPENDING_TRADING: PENDING_TRADING\nBREAK: BREAK';
CREATE TABLE IF NOT EXISTS "aerich" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "version" VARCHAR(255) NOT NULL,
    "app" VARCHAR(20) NOT NULL,
    "content" JSONB NOT NULL
);

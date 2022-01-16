-- upgrade --
ALTER TABLE "asset" ADD "removed_at" TIMESTAMPTZ;
-- downgrade --
ALTER TABLE "asset" DROP COLUMN "removed_at";

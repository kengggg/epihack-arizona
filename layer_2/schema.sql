-- Indexes for Layer 2 / Story 1 queries.
-- Run once against your incidents.db:
--   sqlite3 path/to/incidents.db < layer_2/schema.sql

-- Covers the WHERE clause in story_1.compute_trend:
--   date_of_report BETWEEN ... AND postal_code LIKE :area_prefix || '%'
-- Date comes first because it's the more selective filter (44-day window
-- out of years of data); postal_code as a secondary key lets the LIKE
-- prefix scan stay on the index.
CREATE INDEX IF NOT EXISTS idx_incidents_date_postal
    ON incidents (date_of_report, postal_code);

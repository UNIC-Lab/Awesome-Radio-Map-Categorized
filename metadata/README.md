# Venue Ranking Metadata

`venue-rankings-legacy.csv` provides a venue-level snapshot of the non-empty JCR, Chinese Academy of Sciences (CAS), impact-factor, and CCF values displayed inline in the main paper list.

The original assessment year and source URL were not recorded. The `snapshot_date` column is the migration date, not the year in which a ranking or impact factor was issued. The values are retained to help readers assess venue standing, while this note makes their provenance explicit.

Future ranking updates should:

- record one row per venue rather than repeat values for every paper;
- include an explicit assessment year and a citable source URL;
- distinguish journal metrics from conference rankings; and
- use empty fields for unavailable data instead of the string `None`.

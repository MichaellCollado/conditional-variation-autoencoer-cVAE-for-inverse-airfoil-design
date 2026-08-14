# Where the five seed sections came from

This is source attribution. **It is not a licence grant and does not function
as one.** No licence statement accompanies any of these five files in this
build's inputs, and Table A3 of `PAPER.md` asserts none. Anyone needing
permission to reuse or redistribute any of them should seek it from the source
named below.

The five files are locked inputs. The build does not regenerate them.

| File | Section | Coordinate points | Source |
|---|---|---|---|
| `e387.dat` | Eppler E387 | 61 | UIUC Airfoil Coordinates Database |
| `s1223.dat` | Selig S1223 | 81 | UIUC Airfoil Coordinates Database |
| `sd7003.dat` | SD7003 | 61 | UIUC Airfoil Coordinates Database |
| `seagull.dat` | Seagull, after Liu et al. (2006) | 299 | Constructed for this study, see below |
| `sg6043.dat` | SG6043 | 81 | UIUC Airfoil Coordinates Database |

Point counts are the coordinate rows in each file, which is one fewer than the
line count because the first line of each file is its header name. They match
the counts Table A3 of `PAPER.md` records.

**Two counts appear for the seagull section and both are correct.** Table A3
gives it 299 coordinate points, which is the number of coordinate rows in
`seagull.dat`. Section 2.1.1 measures the reconstruction "over 300 raw
coordinate points", which is the count after `geometry.read_seed_dat` splits the
loop into two surfaces: 150 upper and 150 lower, because the leading edge point
belongs to both and is carried in each. Every seed behaves the same way, at
32 + 30 = 62 against 61 rows for E387 and SD7003, and 46 + 36 = 82 and
44 + 38 = 82 against 81 rows for S1223 and SG6043. The two numbers count
different things and neither is an error.

## The four database sections

Four of the five come from the UIUC Airfoil Coordinates Database, maintained by
Michael Selig at the University of Illinois at Urbana-Champaign. The database
holds approximately 1,650 airfoils at version 2.0, and its archive was last
updated 23 February 2026.

https://m-selig.ae.illinois.edu/ads/coord_database.html

Anyone needing permission to reuse or redistribute these four files should seek
it from the database.

## The avian section

`seagull.dat` is not from that database. It was constructed from the seagull
wing cross section reported in:

> Liu, T., Kuykendoll, K., Rhew, R., and Jones, S. (2006). Avian wing geometry
> and kinematics. AIAA Journal, 44(5), 954 to 963.
> https://doi.org/10.2514/1.16224

Section 2.1.1 of `PAPER.md` describes the construction. The shape comes from
the source's span-averaged coefficients over 2y/b from 0.166 to 0.772, and the
magnitude from the source's envelope equations evaluated at the single station
2y/b = 0.4, because the source splits shape and magnitude that way. Section
2.1.1 also records a correction applied to a conflict inside the source's own
text.

Anyone needing permission to reuse or redistribute this file should seek it
from the article and its publisher.

The seagull's published analogue is also in this directory. Liu et al. (2006)
report that the seagull and merganser airfoils resemble the S1223 once both are
brought to a common maximum camber and thickness, which is why `PAPER.md`
records at section 4.6 that avian-like geometry enters the training
distribution of both arms twice.

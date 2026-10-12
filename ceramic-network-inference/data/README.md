# Data

All data come from the appendices of

> Kroon, E.J. (2024). *Serial Learners: Interactions between Funnel Beaker West and Corded Ware Communities in the Netherlands during the Third Millennium BCE from the Perspective of Ceramic Technology.* Sidestone Press, Leiden. [doi:10.59641/4e367hq](https://doi.org/10.59641/4e367hq)
>
> Appendices: DANS Data Station Archaeology, [doi:10.17026/AR/FAXIOW](https://doi.org/10.17026/AR/FAXIOW), licensed **CC-BY-4.0**.

## `source/` — verbatim files from the deposit

| file | content |
|---|---|
| `Appendix_C_Macroscopy_Paths.csv` | one chaîne opératoire per vessel, with vessel ID and body of knowledge (BoK) |
| `Appendix_C_Macroscopic_Data.xlsx` | vessel descriptions: site ID, phase, vessel type, macroscopic observations |
| `Appendix_B_Sites_Table.csv` | sites: name, region, site type, coordinates (Latin-1, `;`, decimal commas) |
| `nodes_list.csv` | the 49 techniques of Kroon's network and their hydric tolerance (W/L/D/F) |
| `link_list.csv` | the 573 allowed transitions between techniques |

## `source/dans/` — the rest of the deposit

`python scripts/download_dans.py` downloads every file of the deposit except the photographs (Appendix D, 13 tar files, ~36 GB) and the high-resolution figures of the book (~0.6 GB). It keeps DANS's folder structure and checks each file against the checksum DANS publishes (`MANIFEST.csv`). That is 42 files and 36 MB, of which 27 MB is the petrographic reports PDF. Add `--all` for the photographs and figures. The five files in `source/` are byte-identical to their copies here (version 2.0 of the deposit, 2024-10-03).

| folder | content |
|---|---|
| `Appendix_A_Radiocarbon_dates/` | radiocarbon dates (CSV: laboratory number, age BP, error, dated material, context, site, BoK group, typochronology), Bayesian chronological models and bibliography (PDF) |
| `Appendix_B_Site_Information/` | the site table and the site bibliography (PDF) |
| `Appendix_C_Macroscopy_data/` | vessel descriptions and chaînes opératoires (the files in `source/`) |
| `Appendix_D_Photographic_Documentation/` | the list of photographs per vessel (PDF; the photographs themselves are skipped) |
| `Appendix_E_Petrographic_Data/` | thin section ↔ vessel correspondence with fabric groups (CSV), petrographic reports (PDF) |
| `Appendix_F_Simulation/` | simulated chains of sister groups and outgroups: FBN, Odisha, Vlaardingen, random (CSV), and their description (PDF) |
| `Code/` | Kroon's Pathfinder code (`pathfinder.py`, `module_*.py`, `README.txt`) and its inputs and outputs in `ceramics/` (paths per BoK, link lists of the simulated groups) |

## `source/intcal20/` — the radiocarbon calibration curve

`intcal20.14c`: the IntCal20 Northern Hemisphere calibration curve (Reimer et al. 2020, *Radiocarbon* 62: 725–757, doi:10.1017/RDC.2020.41), downloaded from <https://intcal.org/curves/intcal20.14c> (SHA-256 `974a66649f2ac8a53e6c99e256b019ac6982f999b12dcf7193162d4c1c09168e`). Used by `chaine.chronology` to calibrate Kroon's dates (notebook 10.8), as Kroon did with OxCal.

## `clean/` — derived by `python scripts/prepare_data.py`

| file | columns |
|---|---|
| `chains.csv` | `vessel_id`, `bok`, `n_steps`, `chain` (steps joined by `" > "`) |
| `vessels.csv` | `vessel_id`, `context_id`, `site_id`, `phase`, `vessel_type`, `petrographic_sample`, `site_name`, `region`, `site_type`, `longitude`, `latitude`, `approximate_position` |
| `techniques.csv` | `node_id`, `technique`, `tolerance` (e.g. `L\|D`) |
| `technique_links.csv` | `source`, `target` (technique names) |

`python scripts/prepare_data.py --download` fetches the source files from DANS again before rebuilding.

## `derived/` — written by the notebook (Section 4.2)

These are the bootstrap pseudo-documents, for reuse. Each pools $g$ vessels of one BoK, drawn with replacement; there are 100 per BoK and the seed is 0. All triplets are kept, so stop triplets can be removed as wanted.

| file | content |
|---|---|
| `pseudo_documents_training_split.csv` | $g$ = 3, 7, 15, from the 224 training vessels of the notebook's split (fold 1 of seed 0); its 52 held-out vessels remain valid test data |
| `pseudo_documents_all_vessels.csv` | $g$ = 7, from all 276 vessels, like the legacy `bootstrap_documents_*.csv`; for description only, since no vessel is left to test on |

Columns: `source`, `g`, `seed`, `pseudo_document`, `bok`, `vessels` (the $g$ vessel IDs, `;`-separated, repeats possible), `n_triplets`, `triplets` (`|`-separated).

## Labels

| BoK | meaning | vessels | sites |
|---|---|---|---|
| CW | Corded Ware | 89 | 65 |
| EFG | Funnel Beaker West, early phase | 62 | 8 |
| MI | Funnel Beaker West, middle phase | 95 | 14 |
| LFG | Funnel Beaker West, late phase | 30 | 7 |

`FBW.csv` of the earlier notebooks is exactly EFG + MI + LFG. Note that MI vessels come almost entirely from megalithic tombs and EFG/LFG mostly from flat-grave cemeteries, so phase and burial context are confounded. Kroon's Appendix A (Bayesian model 4) defines EFG and LFG as *early* and *late flat graves*: graves with charred wooden structures and a broad spectrum of vessels, and cremation burials with large, complex bowls. His dates place them before CW and beside CW respectively (notebook 10.8).

## Known quirks (reported by `prepare_data.py`)

- Vessel 79 (CW, context "Graf VII no 3") belongs to site 39, which is missing from the site table and from every other file of the deposit; its site metadata are empty.
- Site type `unknown` is Kroon's own label, not a join error. It covers 20 sites, 15 of which supply 20 CW vessels: museum-collection pieces (inventory numbers from 1880 to 1962) without a recorded find context. Three of these sites (21, 66, 130) also have no coordinates.
- The transition `Application separate element → Scraping` (12 vessels) is absent from `link_list.csv`; the grammar in `chaine.data.load_grammar` adds transitions observed in the data.

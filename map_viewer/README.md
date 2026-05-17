# Map Viewer

Small CLI for extracting presentation-ready board snapshots from saved `lmvsgame.json` files.

## Examples

List phases and supply-center counts:

```bash
.venv/bin/python map_viewer/render_map.py list results/3game_experiment_v3/game1/lmvsgame.json
```

Render a phase to SVG:

```bash
.venv/bin/python map_viewer/render_map.py render \
  results/3game_experiment_v3/game1/lmvsgame.json \
  --phase W1905A \
  --output map_viewer/output/v3_game1_w1905a.svg
```

Render with orders and also create a PNG if `rsvg-convert` is installed:

```bash
.venv/bin/python map_viewer/render_map.py render \
  results/3game_experiment_v3/game1/lmvsgame.json \
  --phase F1910M \
  --orders \
  --png \
  --output map_viewer/output/v3_game1_f1910m_orders.svg
```

Render presentation badges with original/colored model logos and exact model labels:

```bash
.venv/bin/python map_viewer/render_map.py render \
  results/3game_experiment_v3/game1/lmvsgame.json \
  --phase W1904A \
  --badges \
  --png \
  --output map_viewer/output/v3_game1_w1904a_badges.svg
```

The renderer uses the repository's bundled Diplomacy engine and `diplomacy/maps/svg/standard.svg`, so borders, ownership colors, units, and order arrows match the game state rather than a reconstructed map.

Render on the newer planning-map artwork:

```bash
.venv/bin/python map_viewer/render_planning_map.py \
  results/3game_experiment_v3/game1/lmvsgame.json \
  --phase W1904A \
  --output map_viewer/output/planning_v3_game1_germany_peak_w1904a_badges.png \
  --width 2400 \
  --badges \
  --units \
  --header
```

The planning-map renderer is raster-first because `map_viewer/assets/Planning Map (v2).svg` does not expose one named path per province. It uses the visible borders as flood-fill barriers, seeds each province from the printed map labels, and tints recovered regions from the saved phase state.

Render with the Diplicity React classical map data:

```bash
.venv/bin/python map_viewer/render_diplicity_map.py \
  results/3game_experiment_v3/game1/lmvsgame.json \
  --phase W1904A \
  --output map_viewer/output/diplicity_v3_game1_germany_peak_w1904a_badges.png \
  --width 2400 \
  --badges \
  --units \
  --header \
  --keep-svg
```

This uses province-level paths from `johnpooch/diplicity-react` rather than
flood-filling from an image, so land recoloring is deterministic and does not
interact with labels or sea regions. The copied source data lives in
`map_viewer/assets/diplicity/`.

Logo assets live in `map_viewer/assets/logos/`. The badge overlay infers `Power -> model` from the saved game's `state_agents` data where possible.

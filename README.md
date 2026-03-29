# Swarm Object Recognition

A swarm-based algorithm for shape recognition in images using color-following bots.

## How It Works

1. **Initialization**: Bots are placed at every pixel along the image edges (top, bottom, left, right)
2. **Movement**: Each bot moves inward, choosing the adjacent pixel with the closest color value
3. **Path Tracing**: Each bot records its path from edge to edge (or until it can't move further)
4. **Visualization**: All paths are overlaid on the original image, revealing object shapes

The collective paths form streamlines that follow color-consistent regions, effectively outlining objects in the image.

### Continuous Wave Mode

In continuous mode, waves of bots are spawned progressively deeper into the image:

| Wave | Top Edge | Bottom Edge | Left Edge | Right Edge |
|------|----------|-------------|-----------|------------|
| 1 | Row 0 | Row H-1 | Col 0 | Col W-1 |
| 2 | Row 1 | Row H-2 | Col 1 | Col W-2 |
| 3 | Row 2 | Row H-3 | Col 2 | Col W-3 |
| ... | ... | ... | ... | ... |

Each wave only traces **untraced paths** within the same **color band** (controlled by `color_threshold`).

### Edge Directions

| Edge | Direction | Candidates |
|------|-----------|------------|
| Top | Down | down-left, down, down-right |
| Bottom | Up | up-left, up, up-right |
| Left | Right | down-right, right, up-right |
| Right | Left | down-left, left, up-left |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Single wave - bots from all edges (default, paths colored by inverted color values)
python swarm_shape.py your_image.png

# Single wave - with color threshold (bots only follow similar colors)
python swarm_shape.py your_image.png all 30
python swarm_shape.py your_image.png top 50

# Continuous mode - spawn waves until edges are covered
python swarm_shape.py your_image.png all continuous
python swarm_shape.py your_image.png top continuous 5  # max 5 waves

# Continuous mode with color threshold
python swarm_shape.py your_image.png all 30 continuous
python swarm_shape.py your_image.png all 30 continuous 5

# Use fixed green color for all paths (instead of color-by-value)
python swarm_shape.py your_image.png all fixed

# Use non-inverted colors (paths show actual colors)
python swarm_shape.py your_image.png all noinvert

# Combine options
python swarm_shape.py your_image.png all 30 continuous 5 fixed
python swarm_shape.py your_image.png all 30 continuous 5 noinvert
```

### Programmatic Usage

```python
from swarm_shape import process_image, trace_paths, visualize_paths

# Complete pipeline - paths colored by inverted color values (default)
output_image, paths, stats = process_image("input.png", "output/traced.png", edges="all")

# With color threshold - bots only follow pixels within color distance of 30
output_image, paths, stats = process_image("input.png", color_threshold=30)

# Continuous mode - spawn waves until edges are covered
output_image, paths, stats = process_image(
    "input.png", 
    edges="all",
    continuous=True,
    max_waves=10
)

# Continuous mode with color threshold
output_image, paths, stats = process_image(
    "input.png", 
    edges="all",
    continuous=True,
    max_waves=10,
    color_threshold=50
)

# Use fixed green color for all paths
output_image, paths, stats = process_image(
    "input.png", 
    color_by_value=False  # Fixed green color
)

# Use non-inverted colors (paths show actual colors)
output_image, paths, stats = process_image(
    "input.png", 
    invert_color=False  # Non-inverted colors
)

# Or step by step
import cv2
image = cv2.imread("input.png")

# Single wave
paths = trace_paths(image, edges="all")

# With color threshold
paths = trace_paths(image, edges="all", color_threshold=30)

# Continuous waves
paths = trace_paths(image, edges="all", continuous=True, max_waves=10)

# Visualize with inverted path colors (default)
visualize_paths(image, paths, "output.png", color_by_value=True, invert_color=True)

# Visualize with non-inverted colors (paths show actual colors)
visualize_paths(image, paths, "output.png", color_by_value=True, invert_color=False)

# Visualize with fixed green color for all paths
visualize_paths(image, paths, "output.png", color_by_value=False)
```

## Output

- `output/traced_image.png` - The input image with all bot paths overlaid (green lines)
- `output/traced_image_display.png` - A display-ready version for viewing

## Customization

You can customize the visualization:

```python
visualize_paths(
    image, 
    paths, 
    output_path='custom.png',
    path_color=(0, 255, 0),  # BGR: green
    line_thickness=1,
    alpha=0.7  # Transparency
)
```

## Algorithm Details

- **Color Distance**: Euclidean distance in RGB space
- **Movement Options**: 3 directions based on starting edge (see table above)
- **Termination**: When bot reaches the opposite edge or can't move further
- **Edge Options**: `all`, `top`, `bottom`, `left`, `right`, or any combination (e.g., `top,bottom`)
- **Modes**:
  - **Single-wave** (default): One bot per edge pixel, all move simultaneously
  - **Continuous**: After each wave completes, new bots spawn at any uncovered edge pixels; repeats until edges are covered or `max_waves` is reached
- **Color Threshold**: Optional maximum color distance for bot movement
  - `None` (default): Bots always move to the closest color pixel
  - `30-50`: Strict color following - bots stay within very similar colors
  - `50-100`: Moderate tolerance - bots can cross slight color variations
  - `100+`: Lenient - bots can follow broader color regions
- **Path Visualization**:
  - `color_by_value=True` (default): Each path is drawn with its color derived from the traced pixels
  - `invert_color=True` (default): Path colors are inverted for better visibility (red→cyan, blue→yellow, green→magenta)
  - `invert_color=False`: Paths show their actual traced colors
  - `color_by_value=False`: All paths are drawn in a fixed green color

## Example

Given an image with objects of distinct colors:
- Bots starting on an object will tend to stay within that object's color region
- Bots starting on the background will flow around objects
- Path density and direction changes reveal object boundaries

## License

MIT

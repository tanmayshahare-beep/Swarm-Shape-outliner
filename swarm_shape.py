"""
Swarm Algorithm for Shape Recognition in Images

This module implements a swarm-based approach where bots start at every pixel
in the top row and trace paths downward by following color-similar pixels.
The collective paths outline objects in the image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def trace_paths(
    image: np.ndarray,
    edges: str = "all",
    continuous: bool = False,
    max_waves: int = 10,
    color_threshold: float = None
) -> List[List[Tuple[int, int]]]:
    """
    Trace paths from bots initialized at image edges.
    
    Each bot moves inward by choosing the adjacent pixel with the most 
    similar color value. Bots can start from any combination of edges.
    
    In continuous mode, each wave starts one row/column deeper into the image.
    Wave 1 starts at edge (row 0), wave 2 at row 1, wave 3 at row 2, etc.
    Bots only trace untraced paths within the same color band.
    
    Args:
        image: Input image as numpy array (BGR format from OpenCV)
        edges: Which edges to start from. Options:
               - "all": top, bottom, left, right
               - "top": only top row
               - "bottom": only bottom row
               - "left": only left column
               - "right": only right column
               - Any combination: e.g., "top,left", "top+bottom"
        continuous: If True, continuously spawn new waves of bots, each starting
                    one pixel deeper into the image
        max_waves: Maximum number of bot waves to spawn (when continuous=True)
        color_threshold: Maximum color distance allowed for bot movement.
                         If None, bots always move to closest color.
                         If set, bots only move to pixels within this color distance.
        
    Returns:
        List of paths, where each path is a list of (row, col) tuples
    """
    # Convert BGR to RGB for consistent color distance calculation
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image
    
    height, width = img_rgb.shape[:2]
    paths = []
    
    # Track visited pixels (for continuous mode)
    visited = set()
    
    # Parse edges parameter
    edges = edges.lower().replace("+", ",")
    if edges == "all":
        edge_list = ["top", "bottom", "left", "right"]
    else:
        edge_list = [e.strip() for e in edges.split(",")]
    
    # Direction mappings: primary direction and candidate offsets
    direction_candidates = {
        "down": [(-1, 1), (0, 1), (1, 1)],    # down-left, down, down-right
        "up": [(-1, -1), (0, -1), (1, -1)],   # up-left, up, up-right
        "right": [(1, 1), (1, 0), (1, -1)],   # down-right, right, up-right
        "left": [(-1, 1), (-1, 0), (-1, -1)]  # down-left, left, up-left
    }
    
    def get_start_positions(edge_list, wave_offset, visited_set):
        """
        Get starting positions for edges at a given wave offset.
        Each wave starts one pixel deeper into the image.
        """
        start_positions = []
        
        if "top" in edge_list:
            row = wave_offset
            if 0 <= row < height:
                for col in range(width):
                    if (row, col) not in visited_set:
                        start_positions.append((row, col, "down"))
        
        if "bottom" in edge_list:
            row = height - 1 - wave_offset
            if 0 <= row < height:
                for col in range(width):
                    if (row, col) not in visited_set:
                        start_positions.append((row, col, "up"))
        
        if "left" in edge_list:
            col = wave_offset
            if 0 <= col < width:
                for row in range(1, height - 1):  # Exclude corners
                    if (row, col) not in visited_set:
                        start_positions.append((row, col, "right"))
        
        if "right" in edge_list:
            col = width - 1 - wave_offset
            if 0 <= col < width:
                for row in range(1, height - 1):  # Exclude corners
                    if (row, col) not in visited_set:
                        start_positions.append((row, col, "left"))
        
        return start_positions
    
    def trace_single_path(start_row, start_col, direction, visited_set, start_color):
        """Trace a single bot path from start position, following similar colors."""
        path = [(start_row, start_col)]
        row, col = start_row, start_col
        
        offsets = direction_candidates.get(direction, [(0, 1), (0, 0), (0, -1)])
        
        if direction in ["down", "up"]:
            max_steps = height
        else:
            max_steps = width
        
        steps = 0
        local_visited = {(start_row, start_col)}
        
        while steps < max_steps:
            candidates = []
            for dr, dc in offsets:
                new_row = row + dr
                new_col = col + dc
                
                if 0 <= new_row < height and 0 <= new_col < width:
                    if (new_row, new_col) not in visited_set and (new_row, new_col) not in local_visited:
                        candidates.append((new_row, new_col))
            
            if not candidates:
                break
            
            current_color = img_rgb[row, col].astype(float)
            
            # Find the candidate with the closest color value
            best_dist = float('inf')
            best_next = None
            
            for next_row, next_col in candidates:
                next_color = img_rgb[next_row, next_col].astype(float)
                dist = np.sqrt(np.sum((current_color - next_color) ** 2))
                
                # Check if within color threshold (if set)
                if color_threshold is not None and dist > color_threshold:
                    continue
                
                if dist < best_dist:
                    best_dist = dist
                    best_next = (next_row, next_col)
            
            if best_next is None:
                break
            
            row, col = best_next
            local_visited.add((row, col))
            path.append((row, col))
            steps += 1
            
            if direction == "down" and row >= height - 1:
                break
            elif direction == "up" and row <= 0:
                break
            elif direction == "right" and col >= width - 1:
                break
            elif direction == "left" and col <= 0:
                break
        
        return path, local_visited
    
    if continuous:
        # Continuous wave mode: each wave starts one pixel deeper
        wave = 0
        while wave < max_waves:
            start_positions = get_start_positions(edge_list, wave, visited)
            
            if not start_positions:
                print(f"  Wave {wave + 1} (offset={wave}): No valid start positions, stopping.")
                break
            
            print(f"  Wave {wave + 1} (offset={wave}): Spawning {len(start_positions)} bots...")
            
            wave_visited = set()
            for start_row, start_col, direction in start_positions:
                start_color = img_rgb[start_row, start_col].astype(float)
                path, local_visited = trace_single_path(
                    start_row, start_col, direction, 
                    visited | wave_visited, 
                    start_color
                )
                
                if len(path) > 1:  # Only add paths that actually moved
                    paths.append(path)
                    wave_visited.update(local_visited)
            
            visited.update(wave_visited)
            wave += 1
        
        print(f"  Completed {wave} wave(s), {len(visited)} total pixels covered.")
    else:
        # Single wave mode: trace once from all edge positions
        start_positions = get_start_positions(edge_list, 0, visited)
        
        for start_row, start_col, direction in start_positions:
            start_color = img_rgb[start_row, start_col].astype(float)
            path, local_visited = trace_single_path(
                start_row, start_col, direction, 
                visited, 
                start_color
            )
            paths.append(path)
            visited.update(local_visited)
    
    return paths


def visualize_paths(
    image: np.ndarray,
    paths: List[List[Tuple[int, int]]],
    output_path: str = 'output/traced_image.png',
    path_color: Tuple[int, int, int] = None,
    line_thickness: int = 1,
    alpha: float = 0.7,
    color_by_value: bool = True,
    invert_color: bool = True
) -> np.ndarray:
    """
    Draw all traced paths on the image and save/display the result.
    
    Args:
        image: Original input image (BGR format)
        paths: List of paths from trace_paths()
        output_path: Path to save the output image
        path_color: Fixed color for drawing paths (BGR format). 
                    If None and color_by_value=True, uses average path color.
        line_thickness: Thickness of path lines
        alpha: Transparency for overlay (1.0 = opaque paths)
        color_by_value: If True, draw each path with its average color
        invert_color: If True and color_by_value=True, invert the path color
                      (e.g., red paths become cyan, blue becomes yellow)
        
    Returns:
        The image with paths drawn on it
    """
    # Create a copy for drawing
    vis = image.copy()
    
    # Create an overlay for semi-transparent paths
    overlay = vis.copy()
    
    # Draw each path
    for path_idx, path in enumerate(paths):
        points = np.array(path, dtype=np.int32)
        
        # Determine path color
        if color_by_value and path_color is None:
            # Calculate average color of this path
            path_colors = []
            for row, col in path:
                path_colors.append(image[row, col])
            avg_color = np.mean(path_colors, axis=0).astype(int)
            
            # Invert color if requested (BGR format)
            if invert_color:
                draw_color = tuple((255 - avg_color).tolist())
            else:
                draw_color = tuple(avg_color.tolist())
        else:
            draw_color = path_color if path_color else (0, 255, 0)
        
        # Draw the path
        for i in range(1, len(points)):
            pt1 = (points[i-1][1], points[i-1][0])  # (col, row)
            pt2 = (points[i][1], points[i][0])
            cv2.line(overlay, pt1, pt2, draw_color, line_thickness)
    
    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save the result
    cv2.imwrite(output_path, vis)
    print(f"Traced image saved to: {output_path}")
    
    # Display using matplotlib (converts BGR to RGB for display)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(vis_rgb)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_display.png'), dpi=150)
    plt.show()
    
    return vis


def get_path_statistics(paths: List[List[Tuple[int, int]]]) -> dict:
    """
    Calculate statistics about the traced paths.
    
    Args:
        paths: List of paths from trace_paths()
        
    Returns:
        Dictionary containing path statistics
    """
    path_lengths = [len(p) for p in paths]
    
    # Count pixel visits (how many bots passed through each pixel)
    visit_counts = {}
    for path in paths:
        for pixel in path:
            visit_counts[pixel] = visit_counts.get(pixel, 0) + 1
    
    return {
        'num_paths': len(paths),
        'min_length': min(path_lengths),
        'max_length': max(path_lengths),
        'avg_length': np.mean(path_lengths),
        'total_pixels_traced': sum(path_lengths),
        'unique_pixels_visited': len(visit_counts),
        'most_visited_pixel': max(visit_counts, key=visit_counts.get) if visit_counts else None,
        'max_visits': max(visit_counts.values()) if visit_counts else 0
    }


def process_image(
    input_path: str,
    output_path: str = 'output/traced_image.png',
    show_stats: bool = True,
    edges: str = "all",
    continuous: bool = False,
    max_waves: int = 10,
    color_threshold: float = None,
    color_by_value: bool = True,
    invert_color: bool = True
) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], dict]:
    """
    Complete pipeline: load image, trace paths, visualize, and return results.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image
        show_stats: Whether to print path statistics
        edges: Which edges to start bots from ("all", "top", "bottom", "left", "right", or combinations)
        continuous: If True, continuously spawn new waves of bots to cover untraced pixels
        max_waves: Maximum number of bot waves (only used when continuous=True)
        color_threshold: Maximum color distance (RGB Euclidean) for bot movement.
                       Bots only follow pixels within this color distance from their current position.
                       Typical values: 10-50 for strict color following, 50-100 for more lenient.
        color_by_value: If True, draw each path with its average color (paths show actual colors).
                       If False, all paths are drawn in green.
        invert_color: If True and color_by_value=True, invert the path color for better visibility.
                      (e.g., red paths become cyan, blue becomes yellow)
        
    Returns:
        Tuple of (output_image, paths, statistics)
    """
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from: {input_path}")
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Trace all paths
    mode = "continuous" if continuous else "single-wave"
    threshold_str = f", threshold={color_threshold}" if color_threshold else ""
    print(f"Tracing paths from edges: {edges} ({mode} mode{threshold_str})...")
    paths = trace_paths(
        image, 
        edges=edges, 
        continuous=continuous, 
        max_waves=max_waves,
        color_threshold=color_threshold
    )
    
    # Get statistics
    stats = get_path_statistics(paths)
    
    if show_stats:
        print("\n=== Path Statistics ===")
        print(f"Number of paths: {stats['num_paths']}")
        print(f"Path length - Min: {stats['min_length']}, Max: {stats['max_length']}, Avg: {stats['avg_length']:.1f}")
        print(f"Total pixels traced: {stats['total_pixels_traced']}")
        print(f"Unique pixels visited: {stats['unique_pixels_visited']}")
        print(f"Most visited pixel: {stats['most_visited_pixel']} ({stats['max_visits']} visits)")
    
    # Visualize and save
    print("\nGenerating output image...")
    output_image = visualize_paths(image, paths, output_path, color_by_value=color_by_value, invert_color=invert_color)
    
    return output_image, paths, stats


if __name__ == "__main__":
    import sys
    
    # Default input image
    input_image = "input_image.png"
    edges_param = "all"
    continuous_mode = False
    max_waves_param = 10
    color_threshold_param = None
    color_by_value_param = True
    invert_color_param = True
    
    # Allow command-line arguments for input path, edges, mode, and threshold
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    if len(sys.argv) > 2:
        edges_param = sys.argv[2]
    if len(sys.argv) > 3:
        if sys.argv[3].lower() == "continuous":
            continuous_mode = True
        elif sys.argv[3].lower() == "fixed":
            color_by_value_param = False
        elif sys.argv[3].lower() == "noinvert":
            invert_color_param = False
        else:
            try:
                color_threshold_param = float(sys.argv[3])
            except ValueError:
                pass
    if len(sys.argv) > 4:
        if sys.argv[4].lower() == "continuous":
            continuous_mode = True
        elif sys.argv[4].lower() == "fixed":
            color_by_value_param = False
        elif sys.argv[4].lower() == "noinvert":
            invert_color_param = False
        else:
            try:
                max_waves_param = int(sys.argv[4])
            except ValueError:
                pass
    if len(sys.argv) > 5:
        try:
            color_threshold_param = float(sys.argv[5])
        except ValueError:
            pass
    if len(sys.argv) > 6:
        if sys.argv[6].lower() == "fixed":
            color_by_value_param = False
        elif sys.argv[6].lower() == "noinvert":
            invert_color_param = False
    if len(sys.argv) > 7:
        if sys.argv[7].lower() == "fixed":
            color_by_value_param = False
        elif sys.argv[7].lower() == "noinvert":
            invert_color_param = False
    
    try:
        output, paths, stats = process_image(
            input_image, 
            edges=edges_param,
            continuous=continuous_mode,
            max_waves=max_waves_param,
            color_threshold=color_threshold_param,
            color_by_value=color_by_value_param,
            invert_color=invert_color_param
        )
        threshold_display = f", threshold={color_threshold_param}" if color_threshold_param else ""
        color_mode = "color-by-value" if color_by_value_param else "fixed-green"
        if color_by_value_param:
            color_mode += "-inverted" if invert_color_param else "-normal"
        print(f"\nDone! (Edges: {edges_param}, Mode: {'continuous' if continuous_mode else 'single-wave'}{threshold_display}, {color_mode})")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nUsage: python swarm_shape.py [path_to_image] [edges] [threshold|continuous|fixed|noinvert] [max_waves] [threshold] [fixed] [noinvert]")
        print("  edges options: 'all', 'top', 'bottom', 'left', 'right', or combinations like 'top,bottom'")
        print("  threshold: maximum color distance for bot movement (e.g., 30, 50, 100)")
        print("  continuous: enable multi-wave mode")
        print("  fixed: use fixed green color for all paths (default: color-by-value)")
        print("  noinvert: don't invert path colors (default: inverted)")
        print("  max_waves: maximum waves for continuous mode (default: 10)")
        print("\nExamples:")
        print("  python swarm_shape.py my_photo.png")
        print("  python swarm_shape.py my_photo.png all 30")
        print("  python swarm_shape.py my_photo.png top continuous")
        print("  python swarm_shape.py my_photo.png all continuous 5")
        print("  python swarm_shape.py my_photo.png all 30 continuous 5")
        print("  python swarm_shape.py my_photo.png all 30 continuous 5 fixed  # fixed green color")
        print("  python swarm_shape.py my_photo.png all 30 continuous 5 noinvert  # non-inverted colors")

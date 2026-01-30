#!/usr/bin/env python3
"""
Generate an interactive flamechart visualization of GGUF model layer sizes.
Usage: uv run python scripts/layer_flamechart.py [model.gguf] [output.html]
"""

import struct
import sys
import json
from collections import defaultdict
from pathlib import Path


def read_gguf_tensors(path):
    """Parse GGUF file and return list of (name, size_bytes) tuples."""
    tensors = []
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError('Not a GGUF file')

        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

        def read_string():
            length = struct.unpack('<Q', f.read(8))[0]
            return f.read(length).decode('utf-8')

        def read_value(vtype):
            if vtype == 0: return struct.unpack('<B', f.read(1))[0]
            elif vtype == 1: return struct.unpack('<b', f.read(1))[0]
            elif vtype == 2: return struct.unpack('<H', f.read(2))[0]
            elif vtype == 3: return struct.unpack('<h', f.read(2))[0]
            elif vtype == 4: return struct.unpack('<I', f.read(4))[0]
            elif vtype == 5: return struct.unpack('<i', f.read(4))[0]
            elif vtype == 6: return struct.unpack('<f', f.read(4))[0]
            elif vtype == 7: return struct.unpack('<?', f.read(1))[0]
            elif vtype == 8: return read_string()
            elif vtype == 9:
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                return [read_value(arr_type) for _ in range(arr_len)]
            elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
            elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
            elif vtype == 12: return struct.unpack('<d', f.read(8))[0]
            else: raise ValueError(f'Unknown type: {vtype}')

        # Skip metadata
        for _ in range(metadata_kv_count):
            read_string()
            vtype = struct.unpack('<I', f.read(4))[0]
            read_value(vtype)

        # Read tensor info
        tensor_info = []
        for _ in range(tensor_count):
            name = read_string()
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensor_info.append((name, dims, dtype, offset))

        # Sort by offset to calculate sizes
        tensor_info.sort(key=lambda x: x[3])
        f.seek(0, 2)
        file_size = f.tell()

        for i, (name, dims, dtype, offset) in enumerate(tensor_info):
            if i < len(tensor_info) - 1:
                size = tensor_info[i + 1][3] - offset
            else:
                size = file_size - offset
            tensors.append((name, size, dims))

    return tensors


def build_hierarchy(tensors):
    """Build hierarchical tree from flat tensor names."""
    root = {"name": "model", "children": {}, "value": 0}

    for name, size, dims in tensors:
        parts = name.split('.')
        node = root
        node["value"] += size

        for part in parts:
            if part not in node["children"]:
                node["children"][part] = {"name": part, "children": {}, "value": 0}
            node = node["children"][part]
            node["value"] += size

        # Store tensor info at leaf
        node["dims"] = dims
        node["is_leaf"] = True

    def convert_children(node):
        """Convert children dict to list for JSON."""
        if node["children"]:
            node["children"] = [convert_children(c) for c in node["children"].values()]
        else:
            del node["children"]
        return node

    return convert_children(root)


def generate_html(hierarchy, output_path):
    """Generate interactive HTML flamechart."""

    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Model Layer Sizes Flamechart</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            padding: 20px;
            background: #16213e;
            border-bottom: 1px solid #0f3460;
        }
        .header h1 { font-size: 1.5em; margin-bottom: 10px; }
        .header .stats { color: #888; font-size: 0.9em; }
        .controls {
            padding: 10px 20px;
            background: #0f3460;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .controls button {
            padding: 8px 16px;
            background: #e94560;
            border: none;
            color: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
        }
        .controls button:hover { background: #ff6b6b; }
        .controls button:disabled { background: #555; cursor: not-allowed; }
        .breadcrumb {
            padding: 10px 20px;
            background: #1a1a2e;
            font-size: 0.85em;
            color: #888;
        }
        .breadcrumb span {
            cursor: pointer;
            color: #4da8da;
        }
        .breadcrumb span:hover { text-decoration: underline; }
        .chart-container {
            padding: 20px;
            overflow-x: auto;
        }
        .flame-row {
            display: flex;
            width: 100%;
            min-height: 28px;
            margin-bottom: 1px;
        }
        .flame-cell {
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            color: white;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            cursor: pointer;
            border-right: 1px solid rgba(0,0,0,0.2);
            transition: filter 0.1s;
        }
        .flame-cell:hover {
            filter: brightness(1.2);
        }
        .flame-cell.selected {
            outline: 2px solid white;
            outline-offset: -2px;
        }
        .tooltip {
            position: fixed;
            background: #16213e;
            border: 1px solid #0f3460;
            padding: 12px;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            max-width: 400px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .tooltip .name { font-weight: bold; color: #4da8da; margin-bottom: 6px; }
        .tooltip .size { color: #e94560; }
        .tooltip .dims { color: #888; margin-top: 4px; }
        .tooltip .pct { color: #50fa7b; }
        .legend {
            padding: 10px 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            font-size: 0.8em;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .legend-color {
            width: 14px;
            height: 14px;
            border-radius: 2px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Model Layer Sizes Flamechart</h1>
        <div class="stats" id="stats"></div>
    </div>
    <div class="controls">
        <button id="reset-btn" disabled>Reset Zoom</button>
        <span style="color:#888">Click to zoom in, button to zoom out</span>
    </div>
    <div class="breadcrumb" id="breadcrumb">model</div>
    <div class="legend" id="legend"></div>
    <div class="chart-container" id="chart"></div>
    <div class="tooltip" id="tooltip" style="display:none"></div>

<script>
const DATA = ''' + json.dumps(hierarchy) + ''';

const COLORS = {
    'encoder': '#e94560',
    'decoder': '#4da8da',
    'joint': '#50fa7b',
    'preprocessor': '#f1fa8c',
    'layers': '#ff79c6',
    'conv': '#bd93f9',
    'self_attn': '#ffb86c',
    'feed_forward1': '#8be9fd',
    'feed_forward2': '#6272a4',
    'norm': '#44475a',
    'linear': '#ff5555',
    'weight': '#50fa7b',
    'bias': '#f1fa8c',
    'default': '#6272a4'
};

function getColor(name, depth) {
    for (const [key, color] of Object.entries(COLORS)) {
        if (name.includes(key)) return color;
    }
    // Vary by depth for visual distinction
    const hue = (depth * 47 + name.charCodeAt(0) * 13) % 360;
    return `hsl(${hue}, 60%, 45%)`;
}

function formatBytes(bytes) {
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(2) + ' MB';
    if (bytes >= 1e3) return (bytes / 1e3).toFixed(2) + ' KB';
    return bytes + ' B';
}

let currentRoot = DATA;
let history = [];

function renderChart(root) {
    const chart = document.getElementById('chart');
    const tooltip = document.getElementById('tooltip');
    const totalValue = root.value;

    document.getElementById('stats').textContent =
        `Total: ${formatBytes(totalValue)} | ${countTensors(root)} tensors`;

    chart.innerHTML = '';

    // Build levels for flame chart (BFS)
    let levels = [[root]];
    while (true) {
        const lastLevel = levels[levels.length - 1];
        const nextLevel = [];
        for (const node of lastLevel) {
            if (node.children) {
                nextLevel.push(...node.children);
            }
        }
        if (nextLevel.length === 0) break;
        levels.push(nextLevel);
    }

    // Render each level
    levels.forEach((level, depth) => {
        const row = document.createElement('div');
        row.className = 'flame-row';

        level.forEach(node => {
            const pct = (node.value / totalValue) * 100;
            if (pct < 0.1) return; // Skip tiny nodes

            const cell = document.createElement('div');
            cell.className = 'flame-cell';
            cell.style.width = pct + '%';
            cell.style.background = getColor(node.name, depth);
            cell.textContent = pct > 3 ? node.name : '';

            cell.addEventListener('mouseenter', (e) => {
                tooltip.style.display = 'block';
                tooltip.innerHTML = `
                    <div class="name">${getFullPath(node, root)}</div>
                    <div class="size">${formatBytes(node.value)}</div>
                    <div class="pct">${pct.toFixed(2)}% of ${root.name}</div>
                    ${node.dims ? '<div class="dims">dims: [' + node.dims.join(', ') + ']</div>' : ''}
                    ${node.children ? '<div class="dims">' + node.children.length + ' children</div>' : ''}
                `;
            });

            cell.addEventListener('mousemove', (e) => {
                tooltip.style.left = Math.min(e.clientX + 10, window.innerWidth - 420) + 'px';
                tooltip.style.top = (e.clientY + 10) + 'px';
            });

            cell.addEventListener('mouseleave', () => {
                tooltip.style.display = 'none';
            });

            if (node.children && node.children.length > 0) {
                cell.addEventListener('click', () => zoomTo(node));
            }

            row.appendChild(cell);
        });

        chart.appendChild(row);
    });

    updateBreadcrumb();
    document.getElementById('reset-btn').disabled = history.length === 0;
}

function getFullPath(node, root) {
    // Simple path reconstruction
    return node.name;
}

function countTensors(node) {
    if (!node.children) return 1;
    return node.children.reduce((sum, c) => sum + countTensors(c), 0);
}

function zoomTo(node) {
    history.push(currentRoot);
    currentRoot = node;
    renderChart(currentRoot);
}

function resetZoom() {
    if (history.length > 0) {
        currentRoot = history[0];
        history = [];
        renderChart(currentRoot);
    }
}

function updateBreadcrumb() {
    const bc = document.getElementById('breadcrumb');
    let path = [currentRoot.name];
    bc.innerHTML = path.map((p, i) =>
        i === path.length - 1 ? p : `<span onclick="goToLevel(${i})">${p}</span>`
    ).join(' > ');
}

function goToLevel(idx) {
    if (idx === 0) resetZoom();
}

// Render legend
const legend = document.getElementById('legend');
Object.entries(COLORS).slice(0, -1).forEach(([name, color]) => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<div class="legend-color" style="background:${color}"></div>${name}`;
    legend.appendChild(item);
});

document.getElementById('reset-btn').addEventListener('click', resetZoom);

renderChart(DATA);
</script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Flamechart saved to: {output_path}")


def main():
    # Default paths
    model_path = sys.argv[1] if len(sys.argv) > 1 else "weights/model.gguf"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "weights/layer_flamechart.html"

    print(f"Reading GGUF model: {model_path}")
    tensors = read_gguf_tensors(model_path)
    print(f"Found {len(tensors)} tensors")

    total_size = sum(t[1] for t in tensors)
    print(f"Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")

    print("Building hierarchy...")
    hierarchy = build_hierarchy(tensors)

    print("Generating HTML flamechart...")
    generate_html(hierarchy, output_path)


if __name__ == "__main__":
    main()

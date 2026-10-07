#!/usr/bin/env python3
"""
Probe a scene's mesh for the OCR lens's words as a continuous heat overlay.

- needs mesh.ply and pointcloud.zarr under the backend dir, the lens cache in <scene>/semantics
- each frame decodes to word probabilities, then lifts onto the vertices (cached as ocr_lens_vertices.npy)
- unseen vertices (depth test) are grey and score zero; scene terms are every observed top-10 word
- label list: words ranked by probability mass (expected vertex count); a click queries that word
- query: comma-separated words; summed probability draws as heat, faces under min p hidden
- click a vertex to chart its top-10 words; --textured shows texture/mesh.obj, picks stay on mesh.ply

Usage:
    HF_HOME=/workspace/models HF_HUB_OFFLINE=1 python docs/examples/ocr_lens_viewer.py \\
        /workspace/outputs/<scene>/<backend> --port 8080
"""

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
import zarr
from PIL import Image

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.reconstructor import store_rows
from collab_splats.semantics.features.ocr_lens import (
    WordVocab,
    load_decoder,
    load_processor,
    word_probabilities,
    word_vocabulary,
)
from collab_splats.semantics.lifting import lift_features
from collab_splats.viewer import Viewer

logger = logging.getLogger(__name__)


def _probability_maps(
    features: zarr.Array, rows: list[int], decoder: torch.nn.Module, vocab: WordVocab
) -> list[torch.Tensor]:
    """
    Each pointcloud frame's patches decoded to word probabilities, (n_words, H_p, W_p) fp16 on CPU.
    """
    maps = []

    for row in rows:
        # Patch states as rows, (H_p * W_p, D)
        fmap = torch.from_numpy(features[row])
        dim, height, width = fmap.shape
        states = fmap.reshape(dim, -1)
        states = states.T

        # Word probabilities per patch, back onto the patch grid
        blocks = [p.half().cpu() for p, _ in word_probabilities(states, decoder, vocab)]
        probs = torch.cat(blocks)
        probs = probs.T
        maps.append(probs.reshape(-1, height, width))

    return maps


def _lift_onto(
    vertices: np.ndarray, colors: np.ndarray, cloud: PointcloudResult, maps: list[torch.Tensor], chunk: int
) -> np.ndarray:
    """
    Word probabilities lifted onto vertices, chunk at a time; (V, n_words) fp16, unobserved rows zero.

    - vertices have no source pixel, so pixel_indices=None: no fallback, unseen stays zero
    """
    blocks = []

    for start in range(0, len(vertices), chunk):
        stop = start + chunk
        part = replace(cloud, points=vertices[start:stop], colors=colors[start:stop], pixel_indices=None)
        lifted = lift_features(maps.__getitem__, part)
        blocks.append(lifted.half().numpy())
        logger.info("lifted vertices %d/%d", min(stop, len(vertices)), len(vertices))

    return np.concatenate(blocks)


def _chart(title: str, words: list[str], probs: np.ndarray) -> str:
    """
    SVG bar chart of word probabilities, pinned to the viewport's top left.

    - x: words, labels tilted 35 degrees; y: probability on a fixed 0-1 axis
    """
    width, height, left, bottom, top = 420, 230, 34, 80, 24
    plot_h = height - bottom - top
    step = (width - left - 8) / max(len(words), 1)
    parts = [f'<text x="{left}" y="15" fill="#ddd" font-size="12" font-weight="bold">{title}</text>']

    # Gridlines and y ticks every 0.25
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1 - tick)
        parts.append(f'<line x1="{left}" x2="{width - 8}" y1="{y:.1f}" y2="{y:.1f}" stroke="#555" stroke-width="0.5"/>')
        parts.append(
            f'<text x="{left - 4}" y="{y + 3:.1f}" fill="#aaa" font-size="9" text-anchor="end">{tick:.2f}</text>'
        )

    # One bar per word, its label tilted under it
    for i, (word, p) in enumerate(zip(words, probs)):
        x = left + i * step
        bar_h = plot_h * float(p)
        cx = x + step / 2
        base = top + plot_h
        parts.append(
            f'<rect x="{x + 2:.1f}" y="{base - bar_h:.1f}" width="{step - 4:.1f}" height="{bar_h:.1f}" fill="#ff5000"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{base + 10:.1f}" fill="#ddd" font-size="10" text-anchor="end" '
            f'transform="rotate(-35 {cx:.1f} {base + 10:.1f})">{word}</text>'
        )

    svg = f'<svg width="{width}" height="{height}">{"".join(parts)}</svg>'
    return (
        '<div style="position:fixed;top:12px;left:12px;z-index:10;padding:6px;'
        f'background:#1a1b1ee6;border-radius:6px">{svg}</div>'
    )


def main() -> None:
    """
    Serve the mesh with a mass-ranked word list, a word query drawn as heat and a click probe.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("backend_dir", type=Path)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--chunk", type=int, default=131_072, help="vertices lifted per lift_features call")
    parser.add_argument("--model_id", default="llava-hf/llava-v1.6-vicuna-7b-hf")
    parser.add_argument("--textured", action="store_true", help="show texture/mesh.obj over the same surface")
    parser.add_argument("--texture_size", type=int, default=4096, help="displayed texture edge, pixels")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Mesh with its vertex colors; light grey when the mesh has none
    mesh = trimesh.load(args.backend_dir / "mesh.ply", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)

    if mesh.visual.kind == "vertex":
        colors = np.array(mesh.visual.vertex_colors[:, :3])
    else:
        colors = np.full((len(vertices), 3), 200, dtype=np.uint8)

    # Cameras and depth the lift tests visibility against
    cloud = PointcloudResult.load_zarr(
        args.backend_dir / "pointcloud.zarr",
        load_world_points=False,
        load_pixel_indices=False,
    )

    # Scene-level lens cache; its rows follow the images/ store
    scene_dir = args.backend_dir.parent
    cache = scene_dir / "semantics" / "ocr_lens.zarr"
    features = zarr.open(str(cache), mode="r")["features"]
    rows = store_rows(scene_dir / "images", cloud.image_paths)

    # Word vocabulary
    processor = load_processor(args.model_id)
    vocab = word_vocabulary(processor.tokenizer)
    row_of = {word: row for row, word in enumerate(vocab.words)}

    # Full-vocabulary word probabilities per vertex, lifted once and cached beside the mesh
    lift_path = args.backend_dir / "ocr_lens_vertices.npy"

    if lift_path.exists():
        full = np.load(lift_path)
    else:
        decoder = load_decoder(args.model_id)
        maps = _probability_maps(features, rows, decoder, vocab)
        full = _lift_onto(vertices, colors, cloud, maps, args.chunk)
        del maps, decoder
        np.save(lift_path, full)

    # Unobserved vertices are zero rows and grey
    observed = full.any(axis=1)
    colors[~observed] = 128
    observed_row = np.cumsum(observed) - 1
    logger.info("unobserved vertices: %d/%d (%.1f%%)", (~observed).sum(), len(vertices), 100 * (~observed).mean())

    # Scene terms: every word in some observed vertex's top-10, renormalized over them
    seen_full = full[observed]
    top = [torch.from_numpy(block).float().topk(10, dim=1).indices for block in np.array_split(seen_full, 64)]
    term_rows = np.unique(torch.cat(top).numpy())
    terms = np.array([vocab.words[row] for row in term_rows], dtype=object)
    term_probs = seen_full[:, term_rows].astype(np.float32)
    term_probs /= term_probs.sum(axis=1, keepdims=True)
    del seen_full

    # Photo texture, when asked: the corner-split OBJ of this same mesh, downscaled for display
    textured = None

    if args.textured:
        textured = trimesh.load(args.backend_dir / "texture" / "mesh.obj", process=False)
        material = textured.visual.material
        size = (args.texture_size, args.texture_size)
        material.image = material.image.resize(size, Image.LANCZOS)

    # Scene, probe chart and query box
    viewer = Viewer(port=args.port)
    viewer.server.gui.configure_theme(control_layout="fixed")
    viewer.add_mesh("mesh", vertices, faces, colors, textured=textured)
    panel = viewer.server.gui.add_html("")
    query = viewer.server.gui.add_text("Query", initial_value="")
    min_prob = viewer.server.gui.add_slider("Query min p", min=0.0, max=1.0, step=0.01, initial_value=0.3)
    search_button = viewer.server.gui.add_button("Search")
    query_note = viewer.server.gui.add_markdown("")
    marker_radius = 0.005 * float(np.linalg.norm(np.ptp(vertices, axis=0)))

    def search(_=None) -> None:
        """
        Draw the summed raw probability of the query words as heat; faces under min p hidden.
        """
        words = [word.strip().lower() for word in query.value.split(",") if word.strip()]
        known = [word for word in words if word in row_of]
        unknown = [word for word in words if word not in row_of]
        query_note.content = f"not in vocabulary: {', '.join(unknown)}" if unknown else ""

        # Empty query clears; unobserved rows are zero so never reach min p
        score = None

        if known:
            columns = [row_of[word] for word in known]
            score = full[:, columns].astype(np.float32).sum(axis=1)

        viewer.show_heat("mesh", score, min_prob.value)

    def select(word: Optional[str]) -> None:
        """
        Put a label-list word in the query box (Clear empties it) and search.
        """
        query.value = word or ""
        search()

    def show(vertex: int) -> None:
        """
        Mark the clicked vertex and chart its top-10 word probabilities in the top left.
        """
        viewer.server.scene.add_icosphere(
            "/probe", radius=marker_radius, color=(255, 0, 255), position=vertices[vertex]
        )

        if not observed[vertex]:
            panel.content = _chart(f"vertex {vertex}: unobserved", [], np.zeros(0))
            return

        values = term_probs[observed_row[vertex]]
        top = np.argsort(-values)[:10]
        panel.content = _chart(f"vertex {vertex}", list(terms[top]), values[top])

    # Words ranked by probability mass: expected vertex count, the total of a click's heat
    mass = full.sum(axis=0, dtype=np.float64)
    viewer.add_label_list("mesh", vocab.words, mass, select)

    # Wire the query and clicks; block
    search_button.on_click(search)
    viewer.on_click("mesh", show)
    viewer.serve_forever()


if __name__ == "__main__":
    main()

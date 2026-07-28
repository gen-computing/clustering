// ============================================================================
// Text rendering utilities for the OpenGL overlay.
//
// Text is rendered as small colored dots on a grid. Each character is defined
// by a 4-wide x 5-tall bitmap. '#' = pixel on, ' ' = pixel off.
//
// FONT: Only uppercase letters (A-Z), digits (0-9), period, and colon are
// defined. Unknown characters render as empty (no dots drawn).
//
// BITMAP CONVENTION:
//   Each character definition is 5 strings of 4 characters, concatenated.
//   Row 0 = top, Row 4 = bottom. Column 0 = left, Column 3 = right.
//   Y is flipped in get_char_bitmap: row 0 in font → y=4.0 in output.
//
//   Example: '1' bitmap
//     "  # "
//     " ## "
//     "  # "
//     "  # "
//     " ###"
//
// COLOR PALETTE: 10 distinct colors for up to 10 clusters.
//   palette[c][0]=R, palette[c][1]=G, palette[c][2]=B, all in [0,1].
//   If clusters > 10: colors repeat (modulo 10).
// ============================================================================

#pragma once

#include <string>
#include <vector>
#include <utility>

// 10-color palette for cluster visualization.
// Chosen for maximum visual distinction at a distance.
extern const float palette[][3];

// build_text_quads(): Generate vertex data for a text string at screen (x,y).
// Each character is built from quads (2 triangles = 6 vertices).
// Returns flat array: {posX, posY, uvU, uvV, r, g, b} repeated per vertex.
// Parameters: text (string to render), x,y (bottom-left start), scale (pixel multiplier), r,g,b (color).
std::vector<float> build_text_quads(const std::string& text, float x, float y, float scale, float r, float g, float b);

// get_char_bitmap(): Extract (col, row) positions of '#' pixels from the font.
// Returns list of coordinates. Row is flipped (font row 0 → y=4.0 in output).
// Unknown characters return empty vector.
std::vector<std::pair<float,float>> get_char_bitmap(char c);

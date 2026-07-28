// ============================================================================
// text.cpp -- Color palette, bitmap font definitions, text rendering utilities.
//
// The palette provides 10 visually distinct colors. The bitmap font provides
// a minimal character set sufficient for rendering clustering metrics.
//
// HOW TEXT RENDERING WORKS:
//   1. For each character in the text string, get_char_bitmap() returns a list
//      of (col, row) dot positions from the 4x5 font grid.
//   2. For each dot, we place a small quad (2 triangles) at the screen position.
//   3. Each quad is rendered with the point shader (using gl_PointCoord).
//   4. The fragment shader's smoothstep discards corners, creating a circular dot.
//   5. The orthographic projection maps pixel coordinates to the screen.
//
// LIMITATIONS:
//   - Only uppercase A-Z, digits 0-9, period, colon. No lowercase, no Unicode.
//   - Fixed 4x5 grid size. Characters are always the same width.
//   - No kerning, ligatures, or typographic features.
//   For production use, integrate freetype or stb_truetype for proper text.
// ============================================================================

#include "text.h"
#include <string>
#include <vector>

// ============================================================================
// COLOR PALETTE
//
// 10 distinct RGB colors, each in [0,1] range for OpenGL.
// Order: Red, Blue, Green, Yellow, Purple, Cyan, Orange, Indigo, Chartreuse, Deep Blue.
//
// Used in the renderer like: palette[cluster_id % 10]
// Colors repeat for clusters beyond index 9 (modulo wrapping).
// ============================================================================
const float palette[][3] = {
    {0.95f, 0.30f, 0.30f},  // 0: Red
    {0.30f, 0.65f, 0.95f},  // 1: Blue
    {0.30f, 0.90f, 0.40f},  // 2: Green
    {0.95f, 0.85f, 0.25f},  // 3: Yellow
    {0.90f, 0.35f, 0.90f},  // 4: Purple
    {0.25f, 0.90f, 0.90f},  // 5: Cyan
    {0.95f, 0.60f, 0.25f},  // 6: Orange
    {0.65f, 0.35f, 0.95f},  // 7: Indigo
    {0.55f, 0.85f, 0.30f},  // 8: Chartreuse
    {0.35f, 0.35f, 0.95f},  // 9: Deep Blue
};

// ============================================================================
// build_text_quads() -- Generate vertex data for a complete text string.
//
// Each character is laid out horizontally with spacing.
// Each character is rendered as 2 triangles (6 vertices × 7 floats per vertex = 42 floats).
//
// Parameters:
//   text:  the string to render (ASCII, uppercase known characters only).
//   x, y:  starting screen position (bottom-left of first character).
//   scale: pixel size multiplier for character dimensions.
//   r,g,b: RGB color for all characters in the string.
//
// Returns: flat float array ready for glBufferData().
//   Format: {posX, posY, uvU, uvV, r, g, b} × (6 * number_of_characters)
// ============================================================================
std::vector<float> build_text_quads(const std::string& text, float x, float y, float scale, float r, float g, float b) {
    std::vector<float> verts;
    float cx = x;
    float char_w = 8.0f * scale;    // Width of each character quad
    float char_h = 12.0f * scale;   // Height of each character quad
    float spacing = 1.0f * scale;   // Extra horizontal space between characters

    for (char ch : text) {
        // Newline: reset X to start, move Y down by character height.
        if (ch == '\n') {
            cx = x;
            y -= char_h + spacing;
            continue;
        }

        // Quad corners: bottom-left (x0,y0), top-right (x1,y1).
        float x0 = cx, y0 = y;
        float x1 = cx + char_w, y1 = y + char_h;

        // 6 vertices = 2 triangles: (bl,br,tr) and (bl,tr,tl)
        // Each vertex: 2 pos + 2 uv + 3 color = 7 floats. 6 * 7 = 42 floats total.
        float q[] = {
            x0,y0, 0,0, r,g,b,   // Bottom-left
            x1,y0, 1,0, r,g,b,   // Bottom-right
            x1,y1, 1,1, r,g,b,   // Top-right
            x0,y0, 0,0, r,g,b,   // Bottom-left (triangle 2)
            x1,y1, 1,1, r,g,b,   // Top-right   (triangle 2)
            x0,y1, 0,1, r,g,b,   // Top-left    (triangle 2)
        };
        verts.insert(verts.end(), q, q + 42);  // Append 42 floats

        cx += char_w + spacing;  // Advance cursor for next character
    }
    return verts;
}

// ============================================================================
// get_char_bitmap() -- Extract dot positions from a character's 4x5 font grid.
//
// Font definitions: each character is 5 strings of 4 characters.
// Row index 0 = top of character, Row 4 = bottom.
// Column index 0 = left, Column 3 = right.
//
// The Y coordinate is FLIPPED in the output:
//   Font row 0 (top) → output Y = 4.0 (highest)
//   Font row 4 (bottom) → output Y = 0.0 (lowest)
// This is because screen coordinates have Y increasing upward.
//
// Parameters:
//   c: ASCII character. Only 0-9, A-Z, space, period, colon are defined.
//
// Returns: list of (column, row) float pairs representing active pixels.
//   Empty list for undefined characters.
// ============================================================================
std::vector<std::pair<float,float>> get_char_bitmap(char c) {
    // Static font data: one entry per ASCII code (only defined characters filled in).
    static const char* font[128] = {};

    // Digits (0-9): 5 lines x 4 chars per character.
    font[' '] = "    " "    " "    " "    " "    ";
    font['0'] = " ## " "#  #" "#  #" "#  #" " ## ";
    font['1'] = "  # " " ## " "  # " "  # " " ###";
    font['2'] = " ## " "#  #" "  # " " #  " "####";
    font['3'] = " ## " "   #" " ## " "   #" " ## ";
    font['4'] = "#  #" "#  #" "####" "   #" "   #";
    font['5'] = "####" "#   " "### " "   #" "### ";
    font['6'] = " ## " "#   " "### " "#  #" " ## ";
    font['7'] = "####" "   #" "  # " " #  " " #  ";
    font['8'] = " ## " "#  #" " ## " "#  #" " ## ";
    font['9'] = " ## " "#  #" " ###" "   #" " ## ";

    // Uppercase letters used in "CLUSTERING ENGINE" and metric labels.
    font['C'] = " ## " "#   " "#   " "#   " " ## ";
    font['L'] = "#   " "#   " "#   " "#   " "####";
    font['U'] = "#  #" "#  #" "#  #" "#  #" " ## ";
    font['S'] = " ## " "#   " " ## " "   #" " ## ";
    font['T'] = "####" "  # " "  # " "  # " "  # ";
    font['E'] = "####" "#   " "### " "#   " "####";
    font['R'] = "### " "#  #" "### " "# # " "#  #";
    font['I'] = " ###" "  # " "  # " "  # " " ###";
    font['N'] = "#  #" "## #" "# ##" "#  #" "#  #";
    font['G'] = " ###" "#   " "# ##" "#  #" " ###";
    font['A'] = " ## " "#  #" "####" "#  #" "#  #";
    font['P'] = "### " "#  #" "### " "#   " "#   ";
    font['O'] = " ## " "#  #" "#  #" "#  #" " ## ";
    font['D'] = "##  " "#  #" "#  #" "#  #" "##  ";
    font['F'] = "####" "#   " "### " "#   " "#   ";
    font['H'] = "#  #" "#  #" "####" "#  #" "#  #";
    font['M'] = "#  #" "## ##" "# ##" "#  #" "#  #";
    font['W'] = "#  #" "#  #" "# ##" "## #" "#  #";
    font['B'] = "### " "#  #" "### " "#  #" "### ";
    font['K'] = "#  #" "# # " "##  " "# # " "#  #";
    font['Y'] = "#  #" "#  #" " ## " "  # " "  # ";
    font['V'] = "#  #" "#  #" "#  #" " #  " "  # ";
    font['X'] = "#  #" " #  " "  # " " #  " "#  #";
    font['Z'] = "####" "   #" "  # " " #  " "####";
    font['Q'] = " ## " "#  #" "#  #" "# # " " ## ";
    font['J'] = "  ##" "   #" "   #" "#  #" " ## ";

    // Punctuation.
    font['.'] = "    " "    " "    " "    " "  # ";
    font[':'] = "    " "  # " "    " "  # " "    ";

    // Copy digit font data so '0'+0 uses font['0'], '0'+1 uses font['1'], etc.
    // This makes indexing by character code work for digits.
    font['0'+0] = font['0'];
    font['0'+1] = font['1'];
    font['0'+2] = font['2'];
    font['0'+3] = font['3'];
    font['0'+4] = font['4'];
    font['0'+5] = font['5'];
    font['0'+6] = font['6'];
    font['0'+7] = font['7'];
    font['0'+8] = font['8'];
    font['0'+9] = font['9'];

    // Extract pixel positions from the 4x5 grid.
    if (c >= 0 && c < 128 && font[c]) {
        std::vector<std::pair<float,float>> pts;
        const char* f = font[c];
        // 5 rows (top to bottom), 4 columns (left to right).
        for (int row = 0; row < 5; row++) {
            for (int col = 0; col < 4; col++) {
                if (f[row*4 + col] == '#')
                    // Flip Y: font row 0 ≈ top ≈ high Y in screen coords.
                    pts.push_back({(float)col, 4.0f - (float)row});
            }
        }
        return pts;
    }
    return {};  // Unknown character: no pixels to draw
}

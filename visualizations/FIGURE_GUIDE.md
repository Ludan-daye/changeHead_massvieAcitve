# Academic Figure Guide

Publication-ready figures for MA (Massive Activation) research.

---

## Figure Overview

### Figure 1: MA Source Evidence
**File**: `conclusion/01_ma_source_evidence.png` (86KB)  
**Type**: Grouped bar chart  
**Models**: 5 (GPT-J, BLOOM, Qwen, Falcon, Mistral)

**Key Finding**: MA originates from MLP, not Attention
- MLP activation values are 3-3496× higher than Attention
- All models show consistent pattern
- Ratios annotated above MLP bars

**Figure Elements**:
- X-axis: Model names
- Y-axis: Maximum activation value
- Blue bars: Attention output
- Red bars: MLP output
- Annotations: MLP/Attention ratio

---

### Figure 2: Attention Role
**File**: `conclusion/02_attention_role.png` (82KB)  
**Type**: Bar chart with baseline  
**Models**: 5

**Key Finding**: Attention provides trigger signal, not MA itself
- Negative change: Attention-triggered models (GPT-J: -96%, BLOOM: -98%)
- Positive change: MLP-dominant models (Qwen: +266%)
- Near-zero change: Hybrid models (Falcon: -21%, Mistral: -18%)

**Figure Elements**:
- X-axis: Model names
- Y-axis: MA change percentage after disabling Attention
- Red bars: Negative change
- Green bars: Positive change
- Black baseline at 0%

---

### Figure 3: Token Type Distribution
**File**: `conclusion/03_function_word_trigger.png` (91KB)  
**Type**: Stacked bar chart  
**Models**: 5

**Key Finding**: 79.2% of MA occurs at function words
- Punctuation (red): Primary trigger
- Function words (orange): Secondary trigger
- Whitespace (blue): Structural markers
- Content words (cyan): Minimal contribution (~20%)

**Figure Elements**:
- X-axis: Model names
- Y-axis: Count (Top 50 MA positions)
- Four stacked segments per model
- Legend: Token types

---

### Figure 4: V-Matrix Dependency ⭐
**File**: `conclusion/04_v_matrix_dependency.png` (120KB)  
**Type**: Horizontal bar chart (sorted)  
**Models**: 7 (includes GPT-2, OPT)

**Key Finding**: 6/7 models strongly depend on V-matrix (>50% change)
- Dark red (>80%): Qwen (-99.1%), Mistral (-82.7%)
- Orange (50-80%): Falcon (-78.8%), GPT-J (-70.7%)
- Blue (<50%): BLOOM (-18.8%) - special case

**Figure Elements**:
- X-axis: MA change after V-ablation (%)
- Y-axis: Models (sorted by absolute change)
- Color-coded by dependency strength
- Reference lines at -50% and -80%
- Value labels on bars

---

### Figure 5: Cross-RQ Metrics Heatmap
**File**: `conclusion/05_comprehensive_heatmap.png` (114KB)  
**Type**: Heatmap with annotations  
**Models**: 5

**Key Finding**: Models exhibit distinct MA mechanism signatures

**Matrix Dimensions**:
- Rows: 5 models
- Columns: 4 metrics
  1. |Attention Change|: Attention contribution
  2. MLP/Attn Ratio: MLP dominance
  3. Function Word %: Trigger location
  4. |V-Ablation Change|: V-dependency

**Figure Elements**:
- Color scale: Blue (low) → Red (high), normalized [0,1]
- Value annotations: Original values in each cell
- Colorbar: Normalized scale

---

### Figure 6: Mechanism Classification
**File**: `conclusion/06_mechanism_classification.png` (140KB)  
**Type**: Text diagram (tree structure)  
**Models**: 5

**Key Finding**: Three distinct MA generation mechanisms

**Classification**:

1. **Attention-Triggered** (MA↓ >50% when disabled)
   - Strong V-Dep: GPT-J (-96%, V-71%)
   - Weak V-Dep: BLOOM (-98%, V-19%)

2. **MLP-Dominant** (MA↑ when Attention disabled)
   - Strong V-Dep: Qwen (+266%, V-99%)

3. **Hybrid** (|MA change| <50%)
   - Falcon (-21%, V-79%)
   - Mistral (-18%, V-83%)

---

### Figure 7: BLOOM Special Case
**File**: `conclusion/07_bloom_special_case.png` (209KB)  
**Type**: Three-panel composite  
**Model**: BLOOM only

**Key Finding**: BLOOM uses early generation + residual propagation

**Panel (a)**: Layer Comparison
- Layer 0: Strong V-dependency (-71%)
- Layer 28: Weak V-dependency (-19%)
- Baseline vs V-ablated MA values

**Panel (b)**: Punctuation Correlation
- Comma (,): 0.44 cosine similarity
- Period (.): 0.42
- Newline (\n): 0.38
- MA direction aligns with punctuation embeddings

**Panel (c)**: Mechanism Diagram
- Early Generation (L0): MLP produces MA
- Residual Propagation (L28): Accumulation via residual
- Semantic Alignment: Boundary marking function

---

## Design Specifications

### Typography
- Font family: Serif (Times New Roman, DejaVu Serif)
- Base font size: 11pt
- Axis labels: 12pt, bold
- Tick labels: 10pt
- Legend: 10pt

### Colors (Color-blind friendly)
- Attention: #377eb8 (Blue)
- MLP: #e41a1c (Red)
- Baseline: #4daf4a (Green)
- Ablated: #999999 (Gray)
- Positive change: #4daf4a (Green)
- Negative change: #e41a1c (Red)
- Strong dependency: #d73027 (Dark red)
- Medium dependency: #fc8d59 (Orange)
- Weak dependency: #4575b4 (Blue)

### Layout
- No figure titles (added in manuscript caption)
- Clean spines (top and right removed)
- Grid: Light dashed lines (#CCCCCC, 50% opacity)
- Edge color: #333333
- Line width: 1.2pt (axes), 0.8pt (grid)
- DPI: 300 (publication quality)

### Figure Sizes
- Single panel: 5×4 inches
- Wide panel: 7×4 inches
- Double panel: 10×4 inches
- Multi-panel: 12×3.5 inches

---

## Data Sources

| Figure | Data Files | Models |
|--------|-----------|--------|
| 1 | `RQ2_mlp_source/verification.json` | 5 |
| 2 | `exp1/README.md` | 5 |
| 3 | `MA_POSITION_TOKEN_ANALYSIS.json` | 5 |
| 4 | `exp6/v_ablation_simple.json` | 7 |
| 5 | All RQ data combined | 5 |
| 6 | RQ1 + RQ5 combined | 5 |
| 7 | BLOOM-specific exp6 data | 1 |

---

## Usage in Manuscript

### Figure Captions (Suggested)

**Figure 1**: Comparison of maximum activation values between Attention and MLP outputs across five LLMs. Numbers above bars indicate MLP/Attention ratio. MA originates from MLP layers, with ratios ranging from 3× (GPT-J) to 3496× (Mistral).

**Figure 2**: MA change after disabling all Attention heads. Negative values indicate Attention-triggered models (GPT-J, BLOOM), positive values indicate MLP-dominant models (Qwen), and near-zero values indicate hybrid mechanisms (Falcon, Mistral).

**Figure 3**: Distribution of token types at Top 50 MA positions. Stacked bars show counts for punctuation (red), function words (orange), whitespace (blue), and content words (cyan). On average, 79.2% of MA occurs at non-semantic positions.

**Figure 4**: V-matrix dependency strength across seven models, measured by MA change after V-ablation. Models are sorted by absolute change. Six of seven models show strong dependency (>50% change), with BLOOM as an exception (18.8%).

**Figure 5**: Heatmap of normalized metrics across four research questions. Rows represent models, columns represent metrics: |Attention Change|, MLP/Attention Ratio, Function Word %, and |V-Ablation Change|. Cell values show original (non-normalized) data.

**Figure 6**: Classification of MA generation mechanisms based on Attention contribution and V-dependency. Three categories emerge: Attention-triggered (MA decreases when disabled), MLP-dominant (MA increases), and Hybrid (minimal change).

**Figure 7**: BLOOM special case analysis. (a) V-ablation effect at Layer 0 vs Layer 28. (b) Cosine similarity between MA direction and punctuation tokens. (c) Proposed mechanism: early generation at L0 with strong V-dependency, followed by residual propagation to L28 with weak V-dependency.

---

## File Formats

All figures are saved as:
- Format: PNG
- Resolution: 300 DPI
- Color mode: RGB
- Compression: Optimized

For vector graphics (EPS/PDF), use:
```python
plt.savefig('figure.pdf', format='pdf', bbox_inches='tight')
```

---

## Reproducibility

**Generation script**: `scripts/generate_visualizations_academic.py`

**Requirements**:
- Python 3.8+
- matplotlib 3.5+
- seaborn 0.12+
- numpy 1.21+

**Command**:
```bash
python3 scripts/generate_visualizations_academic.py
```

**Style consistency**: All figures use the same academic style template defined in the script header.

---

*Last updated: 2025-12-11*  
*Figure version: v2.0 (Academic)*

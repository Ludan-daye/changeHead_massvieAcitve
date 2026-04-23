from .load_model import load_llm, load_vit, load_dinov2_linear_head
from .eval_utils import test_imagenet, setup_dinov2_model_for_eval, fix_reg_mean, eval_ppl
from .plot_utils_llm import plot_3d_feat, plot_layer_ax, plot_attn
from .plot_utils_vit import plot_3d_feat_vit, plot_layer_ax_vit
from .load_data import get_data
from .hook import setup_intervene_hook
from .model_utils import (
    enable_custom_block, enable_custom_attention,
    detect_ma_dimensions, get_mlp_down_proj, set_mlp_down_proj,
    get_mlp_up_proj, get_mlp_submodules,
    get_mlp_module_for_hook,
    _is_moe_layer, _moe_effective_down_proj, _moe_effective_up_proj,
    compute_moe_h2_effective,
    project_out_mlp_down_proj, restore_mlp_down_proj,
)
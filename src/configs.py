import configargparse

def parse_config():
    parser = configargparse.ArgumentParser(
                        prog='Multi-View Diffusion',
                        description='Generate texture given mesh and texture prompt',
                        epilog='Refer to https://arxiv.org/abs/2311.12891 for more details')
    # File Config
    parser.add_argument('--config', type=str, required=True, is_config_file=True)
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--mesh_config_relative', action='store_true', help="Search mesh file relative to the config path instead of current working directory")
    parser.add_argument('--output', type=str, default=None, help="If not provided, use the parent directory of config file for output")
    parser.add_argument('--prefix', type=str, default='MVD')
    parser.add_argument('--use_mesh_name', action='store_true')
    parser.add_argument('--timeformat', type=str, default='%d%b%Y-%H%M%S', help='Setting to None will not use time string in output directory')
    # Diffusion Config
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--negative_prompt', type=str, default='oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect.')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance_scale', type=float, default=15.5, help='Recommend above 12 to avoid blurriness')
    parser.add_argument('--seed', type=int, default=0)
    # ControlNet Config
    parser.add_argument('--cond_type', type=str, default='depth', help='Support depth and normal, less multi-face in normal mode, but some times less details')
    parser.add_argument('--guess_mode', action='store_true')
    parser.add_argument('--conditioning_scale', type=float, default=0.7)
    parser.add_argument('--conditioning_scale_end', type=float, default=0.9, help='Gradually increasing conditioning scale for better geometry alignment near the end')
    parser.add_argument('--control_guidance_start', type=float, default=0.0)
    parser.add_argument('--control_guidance_end', type=float, default=0.99)
    parser.add_argument('--guidance_rescale', type=float, default=0.0, help='Not tested')
    # Multi-View Config
    parser.add_argument('--latent_view_size', type=int, default=96, help='Larger resolution, less aliasing in latent images; quality may degrade if much larger trained resolution of networks')
    parser.add_argument('--latent_tex_size', type=int, default=512, help='Originally 1536 in paper, use lower resolution save VRAM')
    parser.add_argument('--rgb_view_size', type=int, default=1536)
    parser.add_argument('--rgb_tex_size', type=int, default=1024)
    parser.add_argument('--camera_azims', type=int, nargs="*", default=[-180, -90, 0, 90], help='Place the cameras at the listed azim angles')
    parser.add_argument('--top_cameras', action='store_true', help='Two cameras added to paint the top surface')
    parser.add_argument('--mvd_end', type=float, default=0.8, help='Time step to stop texture space aggregation')
    parser.add_argument('--mvd_exp_start', type=float, default=0.0, help='Initial exponent for weighted texture space aggregation, low value encourage consistency')
    parser.add_argument('--mvd_exp_end', type=float, default=6.0, help='End exponent for weighted texture space aggregation, high value encourage sharper results')
    parser.add_argument('--ref_attention_end', type=float, default=0.2, help='Lower->better quality; higher->better harmonization')
    parser.add_argument('--shuffle_bg_change', type=float, default=0.4, help='Use only black and white background after certain timestep')
    parser.add_argument('--shuffle_bg_end', type=float, default=0.8, help='Don\'t shuffle background after certain timestep. background color may bleed onto object')
    parser.add_argument('--mesh_scale', type=float, default=1.0, help='Set above 1 to enlarge object in camera views')
    parser.add_argument('--keep_mesh_uv', action='store_true', help='Don\'t use Xatlas to unwrap UV automatically')
    # Logging Config
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--view_fast_preview', action='store_true', help='Use color transformation matrix instead of decoder to log view images')
    parser.add_argument('--tex_fast_preview', action='store_true', help='Use color transformation matrix instead of decoder to log texture images')
    # Ctrl-Adapter Config
    parser.add_argument(
        "--num_inference_steps",
        type=int, default=50,
        help="We recommend setting the number of inference steps as the same default value of corresponding image/video generation backbone"
        )
    parser.add_argument(
        "--n_sample_frames",
        type=int, default=7,
        help="This is the number of output frames of the video generation model. \
                For image generation, this parameter is not used. \
                For video generation, we recommend setting this parameter with the same default value of the corresponding video diffusion backbone."
    )
    parser.add_argument(
        "--width",
        type=int, default=512,
        help="Our current implementation supports generating videos of size 512 * 512 with I2VGen-XL and SVD, \
            and images of size 1024 * 1024 with SDXL"
            )
    parser.add_argument(
        "--height",
        type=int, default=512,
        help="Our current implementation supports generating videos of size 512 * 512 with I2VGen-XL and SVD, \
            and images of size 1024 * 1024 with SDXL"
            )
    parser.add_argument(
        "--sample",
        type=str, default="spider_mam",
        help="file name of the condition image"
    )
    parser.add_argument(
        "--huggingface_checkpoint_folder",
        type=str, default=None,
        help="Choose the checkpoint folder based on the task. (e.g. i2vgenxl_depth, sdxl_canny) \
            All checkpoint folders are listed in this huggingface repo: \https://huggingface.co/hanlincs/Ctrl-Adapter/tree/main \
            If you want to load from a local checkpoint, set --huggingface_checkpoint_folder as None and use --local_checkpoint_path instead. "
            )
    parser.add_argument(
        '--use_size_512',
        default=True, type=bool,
        help="Our framework currently only support image/video generation with this parameter as True."
    )

    options = parser.parse_args()

    return options

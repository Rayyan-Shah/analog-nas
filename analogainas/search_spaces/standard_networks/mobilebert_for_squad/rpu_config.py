def create_rpu_config(g_max=25, tile_size=256, modifier_std=0.05):

    from aihwkit.simulator.configs import InferenceRPUConfig
    from aihwkit.simulator.configs.utils import BoundManagementType, WeightNoiseType
    from aihwkit.inference import PCMLikeNoiseModel

    rpu_config = InferenceRPUConfig()
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    rpu_config.mapping.max_input_size = tile_size
    rpu_config.mapping.max_output_size = tile_size
 
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=g_max)
    rpu_config.remap.type = WeightRemapType.CHANNELWISE_SYMMETRIC
    rpu_config.clip.type = WeightClipType.FIXED_VALUE
    rpu_config.clip.fixed_value = 1.0
 
    rpu_config.modifier.type = WeightModifierType.MULT_NORMAL
    rpu_config.modifier.rel_to_actual_wmax = True
    rpu_config.modifier.std_dev = modifier_std
    rpu_config.forward = IOParameters()
    rpu_config.forward.out_noise = 0.06
    rpu_config.forward.inp_res = 1 / (2**8 - 2)  # 8-bit resolution.
    rpu_config.forward.out_res = 1 / (2**8 - 2)  # 8-bit resolution.
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config

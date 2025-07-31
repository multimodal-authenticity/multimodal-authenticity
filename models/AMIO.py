import torch.nn as nn
from models.subNets import AlignSubNet
from models.A1_MulT_t import A1_MulT_t
from models.A2_MulT_a import A2_MulT_a
from models.A3_MulT_i import A3_MulT_i
from models.A4_MulT_ta import A4_MulT_ta
from models.A5_MulT_ai import A5_MulT_ai
from models.A6_MulT_ti import A6_MulT_ti
from models.A7_MulT_tai import A7_MulT_tai
from models.A8_MulT_tai_Crossmodal_T import A8_MulT_tai_Crossmodal_T
from models.A9_MulT_tai_Crossmodal_A import A9_MulT_tai_Crossmodal_A
from models.A10_MulT_tai_Crossmodal_I import A10_MulT_tai_Crossmodal_I
from models.A11_MulT_tai_Crossmodal_TIA import A11_MulT_tai_Crossmodal_TIA
from models.A12_LF_LSTM_tai import A12_LF_LSTM_tai
from models.A13_EF_LSTM_tai import A13_EF_LSTM_tai
from models.A14_TFN_tai import A14_TFN_tai
from models.A15_LMF_tai import A15_LMF_tai
from models.A16_MFN_tai import A16_MFN_tai


class AMIO(nn.Module):
    """Adaptive Multimodal Integration Operator - A unified interface for multiple multimodal architectures.

    This class serves as a factory and wrapper for different multimodal models, handling:
    1. Model selection based on configuration
    2. Optional feature alignment
    3. Unified forward pass interface

    Args:
        args (dict): Configuration dictionary containing:
            - model_name: Key specifying which model to instantiate
            - need_model_aligned: Bool for whether to perform feature alignment
            - Other model-specific parameters
    """

    # Model registry mapping configuration names to model classes
    MODEL_MAP = {
        # Single modality transformer variants
        'A1_MulT_t': A1_MulT_t,  # Text-only transformer
        'A2_MulT_a': A2_MulT_a,  # Audio-only transformer
        'A3_MulT_i': A3_MulT_i,  # Visual-only transformer

        # Dual modality transformer variants
        'A4_MulT_ta': A4_MulT_ta,  # Text-Audio transformer
        'A5_MulT_ai': A5_MulT_ai,  # Audio-Visual transformer
        'A6_MulT_ti': A6_MulT_ti,  # Text-Visual transformer

        # Full multimodal transformers
        'A7_MulT_tai': A7_MulT_tai,  # Vanilla multimodal transformer
        'A8_MulT_tai_Crossmodal_T': A8_MulT_tai_Crossmodal_T,  # Text-centric crossmodal
        'A9_MulT_tai_Crossmodal_A': A9_MulT_tai_Crossmodal_A,  # Audio-centric crossmodal
        'A10_MulT_tai_Crossmodal_I': A10_MulT_tai_Crossmodal_I,  # Visual-centric crossmodal
        'A11_MulT_tai_Crossmodal_TIA': A11_MulT_tai_Crossmodal_TIA,  # Full crossmodal integration

        # Sequential models
        'A12_LF_LSTM_tai': A12_LF_LSTM_tai,  # Late Fusion LSTM
        'A13_EF_LSTM_tai': A13_EF_LSTM_tai,  # Early Fusion LSTM

        # Tensor-based fusion models
        'A14_TFN_tai': A14_TFN_tai,  # Tensor Fusion Network
        'A15_LMF_tai': A15_LMF_tai,  # Low-rank Multimodal Fusion

        # Memory-based model
        'A16_MFN_tai': A16_MFN_tai  # Memory Fusion Network
    }

    def __init__(self, args):
        """Initialize the selected multimodal model with optional alignment."""
        super(AMIO, self).__init__()

        # Flag for whether input features need alignment
        self.need_model_aligned = args['need_model_aligned']

        # Initialize alignment subnetwork if needed
        if self.need_model_aligned:
            self.alignNet = AlignSubNet(args, 'avg_pool')  # Using average pooling alignment

            # Update sequence lengths if specified in args
            if 'seq_lens' in args.keys():
                args['seq_lens'] = self.alignNet.get_seq_len()

        # Instantiate the selected model
        lastModel = self.MODEL_MAP[args['model_name']]  # Get model class from registry
        self.model_name = args['model_name']  # Store model name for reference
        self.Model = lastModel(args)  # Initialize the actual model

    def forward(self, text_x, audio_x, vision_x, *args, **kwargs):
        """Forward pass with optional feature alignment.

        Args:
            text_x: Text modality features
            audio_x: Audio modality features
            vision_x: Visual modality features
            *args, **kwargs: Additional arguments passed to underlying model

        Returns:
            Model-specific outputs (typically a dictionary with modality outputs)
        """
        # Perform feature alignment if configured
        if self.need_model_aligned:
            text_x, audio_x, vision_x = self.alignNet(text_x, audio_x, vision_x)

        # Pass through the selected model
        return self.Model(text_x, audio_x, vision_x, *args, **kwargs)

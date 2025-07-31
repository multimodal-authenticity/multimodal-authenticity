import torch
import torch.nn as nn


class AlignSubNet(nn.Module):
    def __init__(self, args, mode):
        """
        Alignment Subnetwork to ensure that the sequence lengths of the text, audio, and video features are the same.

        :param args: Contains various arguments such as feature dimensions and sequence lengths for each modality.
        :param mode: The alignment strategy to use. Currently supports 'avg_pool' (average pooling).
        """
        super(AlignSubNet, self).__init__()

        # Ensure the provided mode is valid
        assert mode in ['avg_pool'], "Invalid alignment mode"

        # Input feature dimensions for text, audio, and video modalities
        in_dim_t, in_dim_a, in_dim_v = args.feature_dims

        # Sequence lengths for text, audio, and video
        seq_len_t, seq_len_a, seq_len_v = args.seq_lens

        # Destination sequence length (target alignment length), set to the text length
        self.dst_len = seq_len_t
        self.mode = mode

        # Dictionary of supported alignment methods
        self.ALIGN_WAY = {
            'avg_pool': self.__avg_pool  # Average pooling method
        }

    def get_seq_len(self):
        """
        Return the target sequence length after alignment.
        """
        return self.dst_len

    def __avg_pool(self, text_x, audio_x, video_x):
        """
        Aligns the input sequences by average pooling to match the destination length (self.dst_len).
        It pads sequences if needed and then applies average pooling.

        :param text_x: Text modality tensor of shape (batch_size, seq_len_t, feature_dim_t)
        :param audio_x: Audio modality tensor of shape (batch_size, seq_len_a, feature_dim_a)
        :param video_x: Video modality tensor of shape (batch_size, seq_len_v, feature_dim_v)
        :return: Aligned text, audio, and video tensors with matching sequence lengths.
        """

        def align(x):
            raw_seq_len = x.size(1)  # Original sequence length

            # If the sequence length matches the target length, return it as is
            if raw_seq_len == self.dst_len:
                return x

            # Calculate pooling size and padding requirements
            if raw_seq_len // self.dst_len == raw_seq_len / self.dst_len:
                pad_len = 0
                pool_size = raw_seq_len // self.dst_len
            else:
                pad_len = self.dst_len - raw_seq_len % self.dst_len
                pool_size = raw_seq_len // self.dst_len + 1

            # Pad the input to match the required length for pooling
            pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
            x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, self.dst_len, -1)

            # Apply average pooling across the sequence
            x = x.mean(dim=1)
            return x

        # Align text, audio, and video modalities
        text_x = align(text_x)
        audio_x = align(audio_x)
        video_x = align(video_x)

        return text_x, audio_x, video_x

    def forward(self, text_x, audio_x, video_x):
        """
        Forward pass to align the modalities. If the sequence lengths are already the same, return them directly.
        Otherwise, apply the specified alignment method.

        :param text_x: Text modality tensor.
        :param audio_x: Audio modality tensor.
        :param video_x: Video modality tensor.
        :return: Aligned text, audio, and video tensors.
        """
        # If all sequence lengths are already the same, no alignment is needed
        if text_x.size(1) == audio_x.size(1) == video_x.size(1):
            return text_x, audio_x, video_x

        # Apply the alignment method (avg_pool in this case)
        return self.ALIGN_WAY[self.mode](text_x, audio_x, video_x)

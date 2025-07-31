import os

from module_test import test

if __name__ == '__main__':
    dataset_name = "Authenticity"  # Dataset
    featurePath = r"data/Influencer_Authenticity_data.pkl"  # Path to data

    model_names = ['A1_MulT_t', 'A2_MulT_a', 'A3_MulT_i', 'A4_MulT_ta', 'A5_MulT_ai', 'A6_MulT_ti', 'A7_MulT_tai',
                   'A8_MulT_tai_Crossmodal_T', 'A9_MulT_tai_Crossmodal_A','A10_MulT_tai_Crossmodal_I', 'A11_MulT_tai_Crossmodal_TIA',
                    'A12_LF_LSTM_tai', 'A13_EF_LSTM_tai', 'A14_TFN_tai', 'A15_LMF_tai', 'A16_MFN_tai']

    for model_name in model_names:
        print("\n" + "=" * 50 , f"Testing model: {model_name}","=" * 50 + "\n")
        test(model_name, dataset_name, featurePath)


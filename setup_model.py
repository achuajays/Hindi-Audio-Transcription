from nemo.collections.asr.models import EncDecCTCModelBPE
import nemo.collections.asr as nemo_asr

import torch

# Load model


# Load the pretrained Hindi ASR model
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_hi_conformer_ctc_medium")
asr_model.save_to("model/stt_hi_conformer_ctc_medium.nemo")

print(f"Model {asr_model} downloaded and saved locally as stt_hi_conformer_ctc_medium.nemo")


model = EncDecCTCModelBPE.restore_from("model/stt_hi_conformer_ctc_medium.nemo")

# Set model to evaluation
model.eval()

# Dummy input for tracing
dummy_input = torch.randn(1, 16000 * 10)  # Batch size 1, 10 sec of audio at 16kHz

# Export to ONNX
model.to('cpu')
model.export("model/asr_model_hi.onnx")

#!/usr/bin/env bash
# run_emotion_x86.sh -- emotion-aware twin of run_x86.sh. Identical environment
# and arguments; the only differences:
#   * launches src/niloys/live/imtalker_emotion_student_live.py (which reuses
#     imtalker_try19_student_live.py and adds the emotion adapter)
#   * the EMOTION_* variables below
#
#   EMOTION_LORA_CKPT=/path/to/last.pt ./run_emotion_x86.sh
#   EMOTION_ID=angry PORT=8992 EMOTION_LORA_CKPT=... ./run_emotion_x86.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"; W="$HERE/weights"
[ -s "$HERE/.snap_path" ] && [ -s "$HERE/.python_path" ] || { echo "run ./prepare_x86.sh <hf_token> first"; exit 1; }
PY="${PY:-$(cat "$HERE/.python_path")}"
SNAP="$(cat "$HERE/.snap_path")"
PORT="${PORT:-8992}"   # 8998 is refused by the server; 8991 may hold the base server
export HF_HOME="${HF_HOME:-$HERE/hf}"

if [ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" != "0" ] && ! "$PY" -c 'import hf_transfer' 2>/dev/null; then
  echo "    note: HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is not installed -> disabling it"
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HERE/.cache/triton}" TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HERE/.cache/inductor}"
export DESIGN_A_DIR="$SRC/niloys" IMTALKER_DIR="$SRC/imtalker"
export PYTHONPATH="$SRC/niloys:$SRC/moshi_pkg:${PYTHONPATH:-}"
unset TRY19_DEC_CKPT
export NATIVE_PASSTHROUGH=1
export TRY19_STUDENT_GRAPH="${TRY19_STUDENT_GRAPH:-1}" TRY19_RENDER_FP16="${TRY19_RENDER_FP16:-1}" TRY19_PD_REFINE="${TRY19_PD_REFINE:-0}"
export TRY19_PD="$W/pd_quant_onpol_robert5_plaus2.pt"
export TRY19_MIMI_DEC_FT="$W/mimi_dec_gan_robert5.pt"
export TRY19_MIMI_GRAPH="${TRY19_MIMI_GRAPH:-1}" TRY19_MIMI_PARTS="${TRY19_MIMI_PARTS:-seanet_quant}" TRY19_MIMI_CAPTURE_AT_INIT="${TRY19_MIMI_CAPTURE_AT_INIT:-1}"
export TRY19_CODE_GUARD="${TRY19_CODE_GUARD:-1}"

_int4_user="${TRY19_INT4_G:-}"; _gate_user="${TRY19_DEP_GATING_INT4:-}"
if [ -s "$HERE/.gpu_probe.env" ]; then set -a; . "$HERE/.gpu_probe.env"; set +a; fi
export TRY19_INT4_G="${_int4_user:-${TRY19_INT4_G:-32}}"
export TRY19_DEP_GATING_INT4="${_gate_user:-${TRY19_DEP_GATING_INT4:-1}}"

if [ -s "$W/seanet_w12_fp32.engine" ]; then
  export TRY19_TRT_SEANET="$W/seanet_w12_fp32.engine"
else
  unset TRY19_TRT_SEANET
  echo "note: no weights/seanet_w12_fp32.engine -- using Mimi's PyTorch SEANet decoder"
fi

# ---- emotion ----
: "${EMOTION_LORA_CKPT:?set EMOTION_LORA_CKPT to the adapter checkpoint (train_emotion_lora.ipynb)}"
export EMOTION_LORA_CKPT
export EMOTION_ID="${EMOTION_ID:-1}"
export EMOTION_CONTROL_FILE="${EMOTION_CONTROL_FILE:-$HERE/emotion_control.txt}"
echo "$EMOTION_ID" > "$EMOTION_CONTROL_FILE"

STUDENT_CKPT="${STUDENT_CKPT:-$W/motionM.pt}"
[ -s "$STUDENT_CKPT" ] || STUDENT_CKPT="$W/motion_student_38M.pt"

mkdir -p "$SRC/niloys/live/static" "$HERE/captures"
ln -sfn "$SRC/imtalker/static/assets" "$SRC/niloys/live/static/assets"
ln -sfn "$SRC/niloys/static/index_try21_no_idle_dissolve.html" "$SRC/niloys/live/static/index.html"
echo "x86 EMOTION: id=$EMOTION_ID adapter=$EMOTION_LORA_CKPT port=$PORT student=$(basename "$STUDENT_CKPT")"
exec "$PY" -u "$SRC/niloys/live/imtalker_emotion_student_live.py" \
  --live_module imtalker_native_passthrough \
  --student_ckpt "$STUDENT_CKPT" --distilled_renderer "$W/renderer_narrow.pt" \
  --motion_gain "${GAIN:-1.2}" --dep_q 8 \
  --blend_frames 0 --jump_thresh 0.15 --mouth_close 0.5 \
  --host 0.0.0.0 --port "$PORT" \
  --html_path "$SRC/niloys/static/index_try21_no_idle_dissolve.html" \
  --generator_path "$W/generator_last.ckpt" --renderer_path "$W/renderer.ckpt" \
  --adapter_path "$W/adapter_last.pt" \
  --adapter_type unitalk_last_layer --adapter_num_layers 12 --adapter_dropout 0.0 \
  --adapter_window_mode lookahead --adapter_future_steps 0 \
  --ref_path "$W/ref.jpeg" --wav2vec_model_path "$W/wav2vec2-base-960h" \
  --moshi_root "$W" --mimi_hf_repo nvidia/personaplex-7b-v1 \
  --moshi_weight "$SNAP/model.safetensors" --mimi_weight "$W/mimi.safetensors" \
  --tokenizer "$W/tokenizer_spm_32k_3.model" \
  --text_prompt "${TEXT_PROMPT:-You are Robert, an assistant from RB Labs. You have knowledge about crypto, finance, investment, technology, and general topics. Answer every question completely, descriptively, and with useful detail.}" \
  --voice_prompt "${VOICE:-Robert_5.pt}" --voice_prompt_dir "$W/voices" \
  --enable_moshi_reply --direct_reply_hidden --reply_hidden_steps_per_chunk 1 \
  --audio_chunk_sec 2.0 --wav2vec_sec 2.0 --fm_chunk_frames 2 --helium_deque_size 25 \
  --prebuffer_chunks "${PREBUFFER:-3}" --render_sub_batch 2 --renderer_precision fp32 \
  --frame_q_backpressure 3 --buffer_ms 80 --skip_fm_audio_encoder \
  --assistant_speech_rms_threshold 0.006 --assistant_speech_hold_chunks 1 \
  --motion_ref_blend 0.0 --motion_prior_noise_blend 0.0 \
  --a_cfg_scale 1.24 --nfe 3 --seed 42 --noise_seed 42 --shared_noise \
  --fp32 --tf32 \
  --silence_helium_path "$W/silence_helium_mean.pt" \
  --jpeg_quality 90 --device cuda --reply_audio_gain 1.0 --output_audio_codec opus \
  --capture_dir "$HERE/captures" \
  --blink_motion_path "$W/blink_motion.pt"

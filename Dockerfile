FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

ARG IMAGE_REVISION
LABEL org.opencontainers.image.title="ORO ShoppingBench SFT batch inference" \
      org.opencontainers.image.description="AnyCloud-powered batch inference of ORO's public ShoppingBench Qwen3-4B SFT model" \
      org.opencontainers.image.source="https://github.com/anycloud-sh/oro-shoppingbench-sft" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0 AND CC-BY-4.0" \
      sh.anycloud.upstream.model="oro-ai/qwen3-4b-shoppingbench-sft" \
      sh.anycloud.upstream.model-revision="2fbf90fc867bb7768a9e034fb8680a830a1d6373"

WORKDIR /opt/oro
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY infer.py LICENSE NOTICE ./
COPY examples ./examples
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
ENTRYPOINT ["python", "/opt/oro/infer.py"]

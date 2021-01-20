FROM jilani95/vgg-transfer-learning-env
LABEL MAINTAINER="Jilani Mokrani"
COPY ./main.py .
ENV DATA_DIR=/data
ENV MODEL_DIR=/models
CMD python3 ./main.py
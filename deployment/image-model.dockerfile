FROM tensorflow/serving:2.7.0

WORKDIR /

COPY models.config /models/models.config
COPY kitchenware-final-model /models/kitchenware-model/1/

CMD ["--model_config_file=/models/models.config"]
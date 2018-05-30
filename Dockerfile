FROM python:3.5-alpine

ENV MOUNT /usr/share/model

COPY requirements.txt model_apply.py model.pkl ${MOUNT}/
COPY lib ${MOUNT}/lib

RUN apk add --update --no-cache --virtual=.build-dep g++ openblas lapack-dev gfortran \
    && pip install --no-cache -r ${MOUNT}/requirements.txt \
    && ln -s /usr/include/locale.h /usr/include/xlocale.h \
    && rm -f ${MOUNT}/requirements.txt \
    && apk del .build-dep

RUN apk add --update --no-cache --virtual=.build-dep git make \
    && mkdir /src && cd /src \
    && git clone --recursive --depth 1 https://github.com/dmlc/xgboost \
    && sed -i '/#define DMLC_LOG_STACK_TRACE 1/d' /src/xgboost/dmlc-core/include/dmlc/base.h \
    && sed -i '/#define DMLC_LOG_STACK_TRACE 1/d' /src/xgboost/rabit/include/dmlc/base.h \
    && cd /src/xgboost; make -j4 \
    && cd python-package; python setup.py install \
    && rm -rf /src \
    && apk del .build-dep

WORKDIR ${MOUNT}

ENTRYPOINT ["python", "model_apply.py"]
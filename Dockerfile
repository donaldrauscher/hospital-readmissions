FROM python:3.5-alpine

ENV BASE /usr/share/model

COPY requirements.txt model_apply.py model.pkl ${BASE}/
COPY lib ${BASE}/lib

RUN apk add --update --no-cache --virtual=.build-dep g++ openblas lapack-dev gfortran \
    && ln -s /usr/include/locale.h /usr/include/xlocale.h \
    && cat ${BASE}/requirements.txt | grep -v xgboost >> ${BASE}/requirements.txt \
    && pip install --no-cache -r ${BASE}/requirements.txt \
    && rm -f ${BASE}/requirements.txt
#    && find /usr/local \
#        \( -type d -a -name test -o -name tests \) \
#        -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
#        -exec rm -f '{}' + \
#    && apk del .build-dep

RUN apk add --update --no-cache --virtual=.build-dep git make \
    && mkdir /src && cd /src \
    && git clone --recursive --depth 1 https://github.com/dmlc/xgboost \
    && sed -i '/#define DMLC_LOG_STACK_TRACE 1/d' /src/xgboost/dmlc-core/include/dmlc/base.h \
    && sed -i '/#define DMLC_LOG_STACK_TRACE 1/d' /src/xgboost/rabit/include/dmlc/base.h \
    && cd /src/xgboost && make -j4 \
    && cd python-package && python setup.py install \
    && rm -rf /src \
    && apk del .build-dep

WORKDIR ${BASE}

ENTRYPOINT ["python", "model_apply.py"]
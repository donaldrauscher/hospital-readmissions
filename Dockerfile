FROM python:3.5-alpine

ENV BASE /usr/share/model

COPY requirements.txt model_apply.py model.pkl ${BASE}/
COPY lib ${BASE}/lib

RUN apk add --update --no-cache libstdc++ openblas lapack-dev \
	&& apk add --update --no-cache --virtual=.build-dep g++ gfortran \
    && ln -s /usr/include/locale.h /usr/include/xlocale.h \
    && cat ${BASE}/requirements.txt | grep -v xgboost >> ${BASE}/requirements-new.txt \ 
    && pip install --no-cache -r ${BASE}/requirements-new.txt \
    && find /usr/local \
        \( -type d -a -name test -o -name tests \) \
        -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
        -exec rm -f '{}' + \
    && apk del .build-dep

RUN apk add --update --no-cache libgomp \
	&& apk add --update --no-cache --virtual=.build-dep git make g++ \
    && mkdir /src && cd /src \
    && export XGB_VERSION="$(cat ${BASE}/requirements.txt | grep xgboost | sed 's/xgboost==//')" \
    && git clone --recursive --depth 1 --branch v${XGB_VERSION} https://github.com/dmlc/xgboost \
    && sed -i '/#define DMLC_LOG_STACK_TRACE 1/d' /src/xgboost/dmlc-core/include/dmlc/base.h \
    && sed -i '/#define DMLC_LOG_STACK_TRACE 1/d' /src/xgboost/rabit/include/dmlc/base.h \
    && cd /src/xgboost && make -j4 \
    && cd python-package && python setup.py install \
    && rm -rf /src \
    && rm -f ${BASE}/requirements*.txt \
    && apk del .build-dep

WORKDIR ${BASE}

ENTRYPOINT ["python", "model_apply.py"]
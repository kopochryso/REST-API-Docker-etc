FROM jupyter/scipy-notebook

WORKDIR /home/jovyan/work

# Extra libs (joblib just in case)
RUN pip install --no-cache-dir joblib

# 🔹 Define environment variables used in train.py
ENV MODEL_DIR=/home/jovyan/work/models \
    MODEL_FILE_LDA=lda_model.joblib \
    MODEL_FILE_NN=nn_model.joblib

# 🔹 Create the folder where models will be saved
RUN mkdir -p ${MODEL_DIR}

# 🔹 Copy your files
COPY train.csv ./train.csv
COPY test.csv ./test.csv
COPY train.py ./train.py
COPY inference.py ./inference.py

# 🔹 Run training at build time
RUN python train.py

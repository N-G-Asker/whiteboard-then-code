FROM python:3.10

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# RUN git clone https://huggingface.co/spaces/codeparrot/apps_metric

# To clone specific branch: `git clone -b <branch> <remote_repo>`

RUN git clone -b xval https://github.com/N-G-Asker/apps_metric.git

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='codeparrot/apps', repo_type='dataset', local_dir='codeparrot/apps', local_dir_use_symlinks='auto')"

# RUN python -c "import datasets; data = datasets.load_dataset('codeparrot/apps'); data.save_to_disk('codeparrot/apps')"

COPY . .

RUN python ./clean_generations.py b3_generations_all_100_iters_10_Dec_2023_03.06AM
CMD python ./grading_gens.py b3_generations_all_100_iters_10_Dec_2023_03.06AM_CLEANED.pkl

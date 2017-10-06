import sys
import os
import logging
import zipfile
import subprocess
import json
import pandas
import numpy as np
import pandas as pd
import tempfile
import tqdm


logger = logging.getLogger(__name__)
PAR_ID = 'paragraph_id'
Q_ID = 'question_id'


def quality(data_file, prediction_file):
    from squad import f1_score

    df_solution = pandas.read_csv(data_file, index_col=[PAR_ID,  Q_ID])['answer']
    df_predictions = pandas.read_csv(prediction_file, index_col=[PAR_ID, Q_ID])['answer']

    df_solution, df_predictions = df_solution.align(df_predictions, join='left')
    df_predictions = df_predictions.fillna('')
    score = {
        'f1': np.mean([f1_score(pred, gt) for gt, pred in zip(df_solution.values, df_predictions.values)]),
    }
    return score


# def quality(data_file, prediction_file):
#     from squad import f1_score
#
#     df_solutions = pandas.read_csv(data_file, index_col=[PAR_ID,  Q_ID])[['answer', 'split']]
#     df_predictions = pandas.read_csv(prediction_file, index_col=[PAR_ID, Q_ID])['answer']
#
#     r = []
#     for f, n in [(lambda x: x, 'all'), (lambda x: x[x.split == 0], '0'), (lambda x: x[x.split == 1], '1')]:
#         df_solution, df_prediction = f(df_solutions).answer.align(df_predictions, join='left')
#         df_prediction = df_prediction.fillna('')
#         x = np.mean([f1_score(pred, gt) for gt, pred in zip(df_solution.values, df_prediction.values)])
#         r.append((n, x))
#     return dict(r)


def create_file_without_answer_column(data_file, data_file_without_answer):
    data = pd.read_csv(data_file)
    data[["paragraph_id", "question_id", "paragraph", "question"]].to_csv(data_file_without_answer, header=True, index=False)
    del data


def get_submission_folder(submission_file, tmpdirname, submission_folder):
    if submission_file is not None:
        if not submission_file.endswith(".zip"):
            raise Exception("submission_file '{}' not looks like zip".format(submission_file))
        with zipfile.ZipFile(submission_file, 'r') as zip_file:
            zip_file.extractall(path=tmpdirname)
            return tmpdirname
    return submission_folder


class DockerEvaluator(object):

    def evaluate(self, data_file, submission_file=None, submission_folder=None, **kwargs):
        import docker
        # URGENT: it works well with native and VirtualBox docker daemon, remote cases doesnot work, but you can fix it
        docker_client = docker.from_env()
        with tempfile.TemporaryDirectory() as tmpdirname:
            # TODO: build image
            image = kwargs.get('image', 'sberbank/sdsj-python')
            with tempfile.TemporaryDirectory() as tmpdirname:
                prediction_file = os.path.join(tmpdirname, "result.csv")
                # if zip archive extract to temp directory
                submission_folder = get_submission_folder(submission_file, tmpdirname, submission_folder)
                data_file_without_answer = os.path.join(tmpdirname, "data.csv")
                # prepare data, remove answer column from data for evaluation
                create_file_without_answer_column(data_file, data_file_without_answer)
                # apply
                container = docker_client.containers.create(
                    image,
                    command=["python3", "predict.py"],
                    # command=["ls", "-lha"],
                    network='none',
                    # cpu_quota=100000,
                    # mem_limit='1g',
                    detach=True,
                    environment={"INPUT": "/src/data.csv", "OUTPUT": "/src/result.csv"},
                    working_dir='/src/',
                )

                # putting files into container (data files and
                import tarfile
                import time
                from io import BytesIO
                import ntpath

                tarstream = BytesIO()
                tar = tarfile.TarFile(fileobj=tarstream, mode='w')
                files = []
                for filename in os.listdir(submission_folder):
                    if os.path.isdir(os.path.join(submission_folder, filename)):
                        continue
                    files.append(os.path.join(submission_folder, filename))
                files.append(data_file_without_answer)

                for filename in tqdm.tqdm(set(files)):
                    tar.add(filename, arcname=ntpath.basename(filename))
                tar.close()

                tarstream.seek(0)

                assert container.put_archive(
                    path='/src/',
                    data=tarstream
                )

                # start container
                container.start()

                start_time = time.time()

                # wait for finish
                status_ticks = []
                elapsed_time = time.time() - start_time
                for current_status in container.stats():
                    elapsed_time = time.time() - start_time

                    current_status = json.loads(current_status.decode('utf8'))

                    is_active = current_status.get('pids_stats', {}).get('current', 0) > 0
                    if not is_active:
                        break

                    logger.info('Running container {} for {}'.format(current_status['name'], elapsed_time))

                    if elapsed_time > 10 * 60:
                        logger.info('Killing container {}'.format(current_status['name']))
                        container.kill()
                        return {'error': 'time limit exceeded (killed after {})'.format(elapsed_time)}

                    status_ticks.append(current_status)

                for line in container.logs(stream=True):
                    print(line.decode('utf-8').strip())

                # read result file
                tardata, info = container.get_archive("/src/result.csv")
                file_like_object = BytesIO(tardata.read())
                tar = tarfile.open(fileobj=file_like_object)
                tar.extractall(tmpdirname)

                # calc quality
                return quality(data_file, prediction_file)


class SimplePythonEvaluator(object):

    def evaluate(self, data_file, submission_file=None, submission_folder=None, **kwargs):
        if submission_file is None and submission_folder is None:
            raise Exception("one of submission_file or submission_folder should not be None")
        if submission_file is not None and submission_folder is not None:
            raise Exception("one of submission_file or submission_folder should be None")
        with tempfile.TemporaryDirectory() as tmpdirname:
            prediction_file = os.path.join(tmpdirname, "result.csv")
            # if zip archive extract to temp directory
            submission_folder = get_submission_folder(submission_file, tmpdirname, submission_folder)
            data_file_without_answer = os.path.join(tmpdirname, "data.csv")
            # prepare data, remove answer column from data for evaluation
            create_file_without_answer_column(data_file, data_file_without_answer)
            # apply
            logging.info("cwd={}".format(submission_folder))
            response = subprocess.run(
                ["python3", "predict.py"], stdout=sys.stdout, cwd=submission_folder, timeout=60 * 10,
                env={'INPUT': data_file_without_answer, 'OUTPUT': prediction_file, **os.environ})
            print(response)
            return quality(data_file, prediction_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='test submission')
    parser.add_argument('-z', '--submission_file', default=None, action='store', help='zip archive')
    parser.add_argument('-f', '--submission_folder', default=None, action='store', help='folder with submission files')
    parser.add_argument('-d', '--data_file', action='store', help='data file')
    parser.add_argument('-t', '--type', help='type of evaluator', default="simple")
    evaluators = {'simple': SimplePythonEvaluator(), 'docker': DockerEvaluator()}
    args = parser.parse_args()
    import logging
    print(args)
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    if args.type not in evaluators:
        raise Exception("unknown evaluator type")
    result = evaluators[args.type].evaluate(**vars(args))
    print("=" * 22)
    print(result)

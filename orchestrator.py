from datetime import timedelta

from airflow import DAG

default_args = {
        'owner'      : 'aroufgangsta',
        'retries'    : 1,
        'retry_delay': timedelta(minutes=5)
}

dag = DAG('move_frames', default_args=default_args, schedule_interval='0 * * * *')

# TODO: Build the dag.

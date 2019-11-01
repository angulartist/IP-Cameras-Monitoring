from __future__ import absolute_import

import logging
from datetime import datetime

import apache_beam as beam
import cv2
import numpy as np
import six
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.transforms import window

from ml_processing.model.inference import DetectLabelsFn


class AddTimestampFn(beam.DoFn):
    def process(self, x, **kwargs):
        date, _ = x

        yield window.TimestampedValue(x, datetime.timestamp(
                datetime.strptime(date, '%Y-%m-%d %H:%M:%S')))


class WindowFormatterFn(beam.DoFn):
    def process(self, x,
                win=beam.DoFn.WindowParam,
                tsp=beam.DoFn.TimestampParam):
        yield '%s - %s - %s' % (win, tsp, x)


class KeyIntoWindow(beam.DoFn):
    def process(self, x, win=beam.DoFn.WindowParam):
        yield (win, x)


class DropKey(beam.DoFn):
    def process(self, x):
        _, frame = x

        yield frame


class TransformToNumpyArrayFn(beam.DoFn):
    def process(self, x):
        np_arr = np.frombuffer(x, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        yield frame


def run(argv=None):
    class TemplateOptions(PipelineOptions):
        @classmethod
        def _add_argparse_args(cls, parser):
            parser.add_argument(
                    '--input', default='frames.log')
            parser.add_argument(
                    '--output', default='output.txt')

    options = PipelineOptions(flags=argv)

    options.view_as(SetupOptions).save_main_session = True
    options.view_as(StandardOptions).streaming = True

    # Uncomment this to run the pipeline on the Cloud (Dataflow)
    # options.view_as(StandardOptions).runner = 'DataflowRunner'

    with beam.Pipeline(options=options) as p:
        parsed_frames = \
            (p
             | 'Read From Pub/Sub' >> beam.io.ReadFromPubSub(
                            topic='projects/alert-shape-256811/topics/ml-flow'
                    ).with_output_types(six.binary_type))
        # | 'Add Event Time' >> beam.ParDo(AddTimestampFn()))

        (parsed_frames
         | 'Transform To Numpy Array' >> beam.ParDo(TransformToNumpyArrayFn())
         | 'Apply Fixed Window' >> beam.WindowInto(
                        window.FixedWindows(5))
         # | 'Add Window As Key' >> beam.ParDo(KeyIntoWindow())
         # | 'Group By Key' >> beam.GroupByKey()
         # | 'Drop Key' >> beam.ParDo(DropKey())
         | 'Group Into Batches' >> beam.BatchElements(min_batch_size=49, max_batch_size=50)
         | 'Detect Labels' >> beam.ParDo(DetectLabelsFn())
         | 'Flatten Labels' >> beam.FlatMap(lambda x: x)
         | 'Pair With One' >> beam.Map(lambda x: (x, 1))
         | 'Sum Label Occurrences' >> beam.CombinePerKey(sum)
         # | 'Format with Window and Timestamp' >> beam.ParDo(WindowFormatterFn())
         # | 'Publish Frames' >> beam.io.WriteToPubSub(
         #                topic='projects/alert-shape-256811/topics/ml-flow-out'))
         | 'Just Print' >> beam.Map(lambda x: logging.info(x)))


if __name__ == '__main__':
    logging \
        .getLogger() \
        .setLevel(logging.INFO)
    run()

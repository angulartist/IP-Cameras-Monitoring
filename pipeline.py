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
from apache_beam.transforms.trigger import AccumulationMode, AfterProcessingTime

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
        _, value = x

        yield value


class TransformToNumpyArrayFn(beam.DoFn):
    def process(self, x):
        np_arr = np.frombuffer(x, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        yield [frame]


class ComputeMean(beam.DoFn):
    def process(self, x, num_frames):
        label, occurrences = x
        mean = sum(occurrences) / num_frames

        yield label, mean


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
        windowed = \
            (p
             | 'Read From Pub/Sub' >> beam.io.ReadFromPubSub(
                            topic='projects/alert-shape-256811/topics/ml-flow',
                            timestamp_attribute='timestamp').with_output_types(six.binary_type)
             | 'Transform To Numpy Array' >> beam.ParDo(TransformToNumpyArrayFn())
             | beam.WindowInto(
                            window.FixedWindows(10),
                            trigger=AfterProcessingTime(5),
                            accumulation_mode=AccumulationMode.DISCARDING))

        counted = \
            (windowed
             | 'Add Default Key' >> beam.Map(lambda x: (0, 1))
             | 'Count Num Frames' >> beam.CombinePerKey(sum)
             | 'Drop Default Key' >> beam.ParDo(DropKey()))

        # a = \
        #     (windowed_frames
        #      | 'Add Window As Key' >> beam.ParDo(KeyIntoWindow()))

        # b = (windowed_frames
        #      | 'Group By Key' >> beam.GroupByKey()
        #      | 'Drop Key' >> beam.ParDo(DropKey()))

        (windowed | 'Detect Labels' >> beam.ParDo(DetectLabelsFn())
         | 'Flatten' >> beam.FlatMap(lambda x: x)
         | 'Pair With One' >> beam.Map(lambda x: (x, 1))
         | 'Group For Mean' >> beam.GroupByKey()
         | 'Mboulouté' >> beam.ParDo(ComputeMean(), beam.pvalue.AsSingleton(counted))
         # | 'Sum Label Occurrences' >> beam.CombineValues(MeanCombineFn())
         # | 'Format with Window and Timestamp' >> beam.ParDo(WindowFormatterFn())
         # | 'Publish Frames' >> beam.io.WriteToPubSub(
         #                topic='projects/alert-shape-256811/topics/ml-flow-out'))
         | 'Just Print' >> beam.Map(lambda x: logging.info(x)))


if __name__ == '__main__':
    logging \
        .getLogger() \
        .setLevel(logging.INFO)
    run()

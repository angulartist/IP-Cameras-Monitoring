from __future__ import absolute_import

import logging
from datetime import datetime

import apache_beam as beam
import cv2
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.transforms import window

from deeper import Deeper


class LogParserFn(beam.DoFn):

    def process(self, element):
        date, frame = element.split(' - ')

        yield (date, frame)


class ExtractBase64StringFn(beam.DoFn):

    def process(self, element):
        _, frame = element

        yield frame


class DetectLabelsFn(beam.DoFn):
    model: Deeper

    def setup(self):
        # Should be stored on GCS
        PROTO_PATH = './ml-model/proto.pbtxt'
        MODEL_PATH = './ml-model/frozen_inference_graph.pb'
        logging.info("[ML] Loading the model 🥶")
        net = cv2.dnn.readNetFromTensorflow(MODEL_PATH, PROTO_PATH)
        self.model = Deeper(net, confidence=0.4)

    def process(self, element):
        self.model.detect(element)

        yield self.model.detect(element)


class AddTimestampFn(beam.DoFn):
    def process(self, element, **kwargs):
        date, _ = element

        yield window.TimestampedValue(element, datetime.timestamp(
                datetime.strptime(date, '%Y-%m-%d %H:%M:%S')))


class WindowFormatterFn(beam.DoFn):
    def process(self, element,
                win=beam.DoFn.WindowParam,
                tsp=beam.DoFn.TimestampParam):
        yield '%s - %s - %s' % (win, tsp, element)


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

    # Uncomment this to run the pipeline on the Cloud (Dataflow)
    # options.view_as(StandardOptions).runner = 'DataflowRunner'

    with beam.Pipeline(options=options) as p:
        parsed_frames = \
            (p
             | 'Read Log File' >> beam.io.ReadFromText('./logs/frames.log')
             | 'Parse Log File' >> beam.ParDo(LogParserFn())
             | 'Add Event Time' >> beam.ParDo(AddTimestampFn()))

        (parsed_frames
         # | 'Apply Fixed Window' >> beam.WindowInto(
         #                window.FixedWindows(5))
         | 'Extract Base64 String' >> beam.ParDo(ExtractBase64StringFn())
         | 'Detect Labels' >> beam.ParDo(DetectLabelsFn())
         # | 'Apply Global Window' >> beam.WindowInto(window.GlobalWindows()))
         | 'Format' >> beam.FlatMap(lambda x: x)
         | 'Pair With One' >> beam.Map(lambda x: (x, 1))
         | 'Sum Label Occurrences' >> beam.CombinePerKey(sum)
         | 'Format with Window and Timestamp' >> beam.ParDo(WindowFormatterFn())
         # | 'Write Output' >> beam.io.WriteToText(file_path_prefix='demo',
         #                                         file_name_suffix='.txt', ))
         | 'Print' >> beam.Map(lambda x: logging.info(x)))


if __name__ == '__main__':
    logging \
        .getLogger() \
        .setLevel(logging.INFO)
    run()

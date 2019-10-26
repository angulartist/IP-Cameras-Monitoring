from __future__ import absolute_import

import json
import logging
from datetime import datetime

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.transforms import window

from vision import VisionHelper


class LogParserFn(beam.DoFn):

    def process(self, element):
        date, frame = element.split(' - ')

        yield (date, frame)


class ExtractBase64StringFn(beam.DoFn):

    def process(self, element):
        _, frame = element

        yield frame


class DetectLabelsFn(beam.DoFn):
    vision_helper: VisionHelper

    def setup(self):
        self.vision_helper = VisionHelper()

    def process(self, elements):
        logging.info('About to log %d elements', len(elements))

        yield zip(elements, self.vision_helper.batch_annotate_images(elements))


class AddTimestampFn(beam.DoFn):
    def process(self, element, **kwargs):
        date, _ = element

        yield window.TimestampedValue(element, datetime.timestamp(
                datetime.strptime(date, '%Y-%m-%d %H:%M:%S')))


def format_json(x):
    frame, label = x
    name, all_vertices = label
    bounding_box = [{'x': v.x, 'y': v.y} for v in all_vertices.vertices]

    return json.dumps({
            'frame'       : str(frame),
            'label_name'  : str(name),
            'bounding_box': bounding_box
    })


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
        (p
         | 'Read Log File' >> beam.io.ReadFromText('./logs/frames.log')
         | 'Parse Log File' >> beam.ParDo(LogParserFn())
         # | 'Add Event Time' >> beam.ParDo(AddTimestampFn())
         # | 'Apply Fixed Window' >> beam.WindowInto(
         #                window.FixedWindows(3))
         | 'Extract Base64 String' >> beam.ParDo(ExtractBase64StringFn())
         | 'Group Into Batch' >> beam.BatchElements(min_batch_size=10, max_batch_size=10)
         | 'Detect Labels' >> beam.ParDo(DetectLabelsFn())
         | 'Flatten' >> beam.FlatMap(lambda x: x)
         # | 'Pair With One' >> beam.Map(lambda x: (x, 1))
         # | beam.CombinePerKey(sum)
         | 'Format' >> beam.Map(format_json)
         | 'Write Output' >> beam.io.WriteToText(file_path_prefix='output',
                                                 file_name_suffix='.json', ))
        # | 'Print' >> beam.Map(lambda x: logging.info(x)))


if __name__ == '__main__':
    logging \
        .getLogger() \
        .setLevel(logging.INFO)
    run()

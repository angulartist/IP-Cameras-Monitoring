class VisionHelper(object):

    @staticmethod
    def batch_annotate_images(frames):
        import base64
        from google.cloud import vision
        from google.cloud.vision import enums
        # Initialize AnnotatorClient
        client = vision.ImageAnnotatorClient()
        # Initialize an empty requests array
        requests = []
        for frame in frames:
            # Decode base64 string
            frame = base64.b64decode(frame)
            # Build a request object and add it to the batch
            requests.append({
                    'image'   : {'content': frame},
                    'features': [
                            {
                                    'type': enums.Feature.Type.LOGO_DETECTION,
                            }
                    ],
            })
        # Process label detection
        batch = client.batch_annotate_images(requests)
        # Return back label descriptions
        return \
            [
                    (label.description, label.bounding_poly)
                    for response in batch.responses
                    for label in response.logo_annotations
            ]

def detect_labels(content, encoded=True):
    """
    Detect labels for the given image frame.¬
    :param encoded: Set to False if plain image frame. Default is True
    :param content: Image frame or its base64 string representation
    :return: <nothing>
    """
    import base64
    from google.cloud import vision
    from google.cloud.vision import types
    # Initialize AnnotatorClient
    client = vision.ImageAnnotatorClient()
    # Either use the image itself or the base64string representation
    content = base64.b64decode(content) if encoded else content
    # Define the image to analyze
    image = types.Image(content=content)
    # Process label detection
    response = client.label_detection(image=image)
    # Retrieve annotations
    labels = response.label_annotations

    # Print out labels found
    print('Labels:')
    for label in labels:
        print(label.description)

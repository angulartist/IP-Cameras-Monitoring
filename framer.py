import base64
import logging
import time
from logging.handlers import TimedRotatingFileHandler

import cv2

from vision import detect_labels


def rescale_frame(frame, scale_percent=50):
    """
    Rescale a frame for the given scale percent.
    :param frame: The image frame to rescale.
    :param scale_percent: The scale  percent ratio to apply. Default is 50.
    :return: A new rescaled image frame.
    """
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def init_logger(name='Rotating Log', rollover='H', every=1):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(
            './frames.log',
            when=rollover,
            interval=every)

    logger.addHandler(handler)

    return logger


def main(stream=None):
    frame_rate = 1
    prev = 0
    logger = init_logger()

    stream = cv2.VideoCapture(
            "https://r7---sn-4gxx-25ge7.googlevideo.com/videoplayback?expire=1571872912&ei=MIywXZ-8G-Ho1wayqbCwDQ&ip=2a01%3Ae35%3A8a0a%3A5530%3A796e%3Aade3%3A9878%3A2687&id=o-AAXLUiIbxKabXL-8GsQHtQsHG5lsi24Kw5JWdWyB6P4J&itag=248&aitags=133%2C134%2C135%2C136%2C137%2C160%2C242%2C243%2C244%2C247%2C248%2C278&source=youtube&requiressl=yes&mm=31%2C29&mn=sn-4gxx-25ge7%2Csn-4gxx-25gel&ms=au%2Crdu&mv=m&mvi=6&nh=EAE%2CEAE&pl=45&initcwndbps=450000&mime=video%2Fwebm&gir=yes&clen=76873138&dur=331.364&lmt=1560827590031998&mt=1571851186&fvip=7&keepalive=yes&fexp=23842630&c=WEB&txp=5431432&sparams=expire%2Cei%2Cip%2Cid%2Caitags%2Csource%2Crequiressl%2Cmime%2Cgir%2Cclen%2Cdur%2Clmt&lsparams=mm%2Cmn%2Cms%2Cmv%2Cmvi%2Cnh%2Cpl%2Cinitcwndbps&lsig=AHylml4wRAIgKg_Np6jhesD_CK-6J7ilZqZe779z9QGnChHMgFxA9fICIFd_9zXKsL4QSQB--3chP8VyaOiqv2wfvis2QXf7aYdC&sig=ALgxI2wwRAIgFsAY_x_z9DrLpN4syu2umO07vfjJRKmDInsVktJp_xsCIC1DuHzdH17Oq8MK8fbYfzMUSARYkhIMPNY_cl6JUb89&ratebypass=yes")

    while True:
        time_elapsed = time.time() - prev

        has_frames, frame = stream.read()

        if time_elapsed > 1. / frame_rate and has_frames:
            prev = time.time()

            frame = rescale_frame(frame, scale_percent=30)
            _, buffer = cv2.imencode('.jpg', frame)
            base64string = base64.b64encode(buffer)
            logger.info(base64string)
            detect_labels(base64string)

            cv2.waitKey(1000)

    # Release the stream
    stream.release()


if __name__ == '__main__':
    main()

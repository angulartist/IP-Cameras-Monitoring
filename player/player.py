from collector.utils import PubSubClient


def main():
    sub = PubSubClient(topic_name='ml-flow-out')
    sub.receive()


if __name__ == '__main__':
    main()

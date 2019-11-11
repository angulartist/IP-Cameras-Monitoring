import setuptools

PACKAGE_NAME = 'ml_processing'
PACKAGE_VERSION = '0.0.1'
REQUIRED_PACKAGES = [
        'numpy',
        'opencv-python',
        'imutils',
        'absl'
]

setuptools.setup(
        name=PACKAGE_NAME,
        version=PACKAGE_VERSION,
        description='quick ml experiment',
        install_requires=REQUIRED_PACKAGES,
        packages=setuptools.find_packages(),
        package_data={
                PACKAGE_NAME: [
                        'model/*',
                        'model/trained/*'
                ]
        }
)
